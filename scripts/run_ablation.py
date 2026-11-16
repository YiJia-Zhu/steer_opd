#!/usr/bin/env python3
"""Orchestrate the zero-update overlap phase of ablation; stop after frozen confirm."""

import argparse
import os
import subprocess
import sys
import signal
import time

from steer_opd.compass_campaign import frozen_json, make_base_job, validate_config, scheme_conditions
from steer_opd.compass_job import load_codecs, project_path, model_path, MODEL_ROLES
from steer_opd.execution_policy import execution_policy, require_execution_allowed
from steer_opd.resumable_core import ROOT, atomic_json, digest, read_json
from steer_opd.experiment_runtime import initialize_stage, Progress, child_environment, console_log
from steer_opd.stage_io import metadata_dir, logs_dir


def missing_assets(cfg):
    missing = []
    for role in MODEL_ROLES:
        model = model_path(cfg[role]).resolve()
        index = model / "model.safetensors.index.json"
        if index.exists():
            shards = set(read_json(index)["weight_map"].values())
            if any(not (model / name).is_file() for name in shards):
                missing.append({"role": role, "reason": "missing_weight_shards", "path": cfg[role]})
        elif not (model / "model.safetensors").is_file():
            missing.append({"role": role, "reason": "checkpoint_weights_missing", "path": cfg[role]})
    return missing


def execute(cfg):
    require_execution_allowed(cfg)
    run=project_path(cfg["run_dir"])
    initialize_stage(cfg,run)
    with console_log(project_path(logs_dir(cfg))/"console.log"):
        with Progress(project_path(logs_dir(cfg)),"campaign",7 if cfg.get('stop_after')=='source_overlap' else
                      8+2*(2 if cfg.get('include_teacher_ablation') else 1)+2*len({c['steer_role'] for c in scheme_conditions(cfg)}),unit="stage",eta=False,show_bar=False) as progress:
            atomic_json(run/'campaign.json',{'status':'running','pid':os.getpid(),'attempt_id':progress.attempt_id,
                'experiment':cfg['experiment'],'phase':cfg['phase'],
                'started_at':progress.started_at,'seed':42,'parameter_updates':0,'automatic_opd':False})
            return _execute(cfg,progress)


def stage_runners(cfg,progress):
    run = project_path(cfg["run_dir"])
    engine_python = str(ROOT / cfg["engine_python"]) if cfg.get("engine_python") else sys.executable
    def gpu(name,count,command,*,interpreter=None,stage_output=None,stage=3):
        from run_on_idle_gpus import tee_process,process_ticks
        stage_root=project_path(cfg['step_dirs'][str(stage)]) if cfg.get('layout_version')==1 else run
        logroot=stage_root/'logs' if cfg.get('layout_version')==1 else run
        logroot.mkdir(parents=True,exist_ok=True)
        expected=[interpreter or engine_python,*command]
        progress.phase=name;progress.event("stage_requested",name=name,command=expected)
        env=child_environment(expected[0]);env['STEER_OPD_RUN_ROOT']=str(stage_root)
        # Adopt a live stage after parent interruption instead of admitting it twice.
        for previous in sorted((logroot/'attempts').glob(f'{name}_*'),reverse=True):
            queue_path=previous/'queue.json'
            if not queue_path.exists():continue
            queue=read_json(queue_path)
            if queue.get('command')!=expected or process_ticks(queue['pid'])!=queue.get('start_ticks') or queue.get('start_ticks') is None:continue
            progress.event('adopting_live_stage',name=name,attempt=str(previous),pid=queue['pid'])
            while process_ticks(queue['pid'])==queue['start_ticks']:time.sleep(1)
            marker=previous/'process.json'
            if not marker.exists() or read_json(marker).get('exit_code')!=0: raise RuntimeError(f'Adopted stage did not finish successfully: {previous}')
            progress.event("stage_finished",name=name,cache_reused=True);return
        stage_output=stage_output or (run/'references'/name.removeprefix('reference_') if name.startswith('reference_') else
                      run/'native_scores'/name.removeprefix('native_') if name.startswith('native_') else
                      run/'select' if name=='overlap_select' else run/'confirm' if name=='frozen_confirm' else run/name)
        if stage_output.exists():
            progress.event('cache_validation_start',name=name)
            trial_env={**env,'STEER_OPD_CACHE_ONLY':'1','CUDA_VISIBLE_DEVICES':''}
            with (logroot/f'{name}_cache_validation.log').open('ab') as log:
                trial=subprocess.Popen(expected,cwd=ROOT,env=trial_env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
                code=tee_process(trial,log)
            if code==0:
                progress.event("stage_finished",name=name,cache_reused=True);return
            if code!=75:raise subprocess.CalledProcessError(code,expected)
        attempt=logroot/'attempts'/f'{name}_{time.time_ns()}'
        attempt.mkdir(parents=True,exist_ok=True)
        launch=[sys.executable,str(ROOT/'scripts/run_on_idle_gpus.py'),'--count',str(count),
            '--max-project-gpus','2','--wait-seconds',str(cfg['gpu_wait_seconds']),'--run-dir',str(attempt)]
        if os.environ.get('STEER_OPD_GPU_IDS'):launch+=['--gpu-ids',os.environ['STEER_OPD_GPU_IDS']]
        launch+=['--',*expected]
        with (attempt/'supervisor.log').open('xb') as log:
            child=subprocess.Popen(launch,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=tee_process(child,log)
            except BaseException:
                if child.poll() is None:
                    child.send_signal(signal.SIGTERM)
                    try:child.wait(timeout=40)
                    except subprocess.TimeoutExpired:child.kill();child.wait()
                raise
        if code:raise subprocess.CalledProcessError(code,launch)
        progress.event("stage_finished",name=name)
    def cpu(name,script,config_path,extra=(),*,stage=2):
        from run_on_idle_gpus import tee_process
        folder=project_path(cfg['step_dirs'][str(stage)])/'logs'
        folder.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,script,'--config',str(config_path),*extra]
        env={**child_environment(),'CUDA_VISIBLE_DEVICES':''}
        with (folder/f'{name}.log').open('ab') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=tee_process(child,log)
            except BaseException:
                if child.poll() is None:
                    child.terminate()
                    try:child.wait(timeout=40)
                    except subprocess.TimeoutExpired:child.kill();child.wait()
                raise
        if code:raise subprocess.CalledProcessError(code,command)
    return gpu,cpu


def _execute(cfg,progress):
    from steer_opd.stage_campaign import execute as execute_stages
    gpu,cpu=stage_runners(cfg,progress)
    return execute_stages(cfg,progress,gpu,cpu)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true",help="Reuse committed artifacts; never bypass the user hold")
    args = parser.parse_args()
    cfg = read_json(args.config)
    validate_config(cfg)
    if args.execute:
        require_execution_allowed(cfg)
    run = project_path(cfg["run_dir"])
    frozen_json(project_path(metadata_dir(cfg))/"config.json",cfg)
    missing = missing_assets(cfg)
    codecs = load_codecs(cfg)
    questions = read_json(project_path(cfg['input_data_dir'])/'questions.json') if (project_path(cfg['input_data_dir'])/'questions.json').exists() else []
    policy = execution_policy(cfg)
    held = bool(cfg.get("execution_hold"))
    status = {**policy, "status": "awaiting_user_approval" if held else
              ("prerequisites_pending" if cfg.get("execution_blockers") else
               ("missing_assets" if missing else "ready_for_execution")), "config_hash": digest(cfg),
              "experiment":cfg['experiment'],"phase":cfg['phase'],
              "asset_status": "missing_assets" if missing else "present_not_gpu_validated",
              "screen_backend": cfg["screen_backend"],
              "recorded_metrics": ["overlap", "P_overlap"], "gate_metric": cfg.get("gate_metric", "overlap"),
              "missing_assets": missing, "parameter_updates": 0, "automatic_opd": False,
              "training_questions_prepared": len(questions), "planned_scheme_count": len(make_base_job(cfg,codecs,'unused')['conditions']),
              'executing_sv_schemes':0 if cfg.get('stop_after')=='source_overlap' else len(make_base_job(cfg,codecs,'unused')['conditions']),
              "native_supports": {r: {"mapped_tokens": len(c.valid_ids), "support_size": c.support_size,
                                      "mapping_sha256": c.mapping_sha256} for r, c in codecs.items()},
              'strength_policies':cfg['strength_policies'],
              'stop_after':cfg.get('stop_after','panel'),
              "stages": (["unsteered_source_top16","native_teacher_span_scores","source_overlap_metrics_and_plots","user_review"]
                         if cfg.get('stop_after')=='source_overlap' else ["teacher_texts", "unsteered_source_top16", "student_T_S_activations", "two_T_S_banks",
                         "native_teacher_span_scores", "select_threshold", "fixed_scheme_scores", "frozen_confirm", "user_review"])}
    if cfg.get('include_teacher_ablation') and cfg.get('stop_after')!='source_overlap':
        status['stages']=['teacher_texts','unsteered_source_top16','student_T_S_capture','teacher_S_T_capture',
                          'four_reader_space_banks','native_teacher_span_scores','select_threshold',
                          'student_and_teacher_fixed_prefix_screens','frozen_confirm','user_review']
    atomic_json(project_path(metadata_dir(cfg))/"preflight.json",status)
    if not args.execute:
        print(status)
        return
    if missing:
        raise FileNotFoundError(f"Weights are incomplete; no GPU launched: {missing}")
    try:
        execute(cfg)
    except BaseException as error:
        atomic_json(run / "campaign.json", {"status": "interrupted_resumable" if isinstance(error,(KeyboardInterrupt,SystemExit)) else "failed_resumable", "error": repr(error),
                    "parameter_updates": 0, "automatic_opd": False})
        raise


if __name__ == "__main__":
    main()
