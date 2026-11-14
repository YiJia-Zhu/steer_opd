#!/usr/bin/env python3
"""Complete zero-update T/Ref/S preparation and eight-scheme orchestration."""

import argparse
import os
import subprocess
import sys
import signal
import time

from steer_opd.compass_campaign import (frozen_json, generation_job, make_base_job,
                                        prepare_initial_jobs, prepare_screen_jobs, validate_config)
from steer_opd.compass_job import load_codecs, project_path, model_path
from steer_opd.execution_policy import execution_policy, require_execution_allowed
from steer_opd.resumable_core import ROOT, atomic_json, digest, read_json
from steer_opd.experiment_runtime import initialize_stage, Progress, child_environment, console_log, export_artifacts


def missing_assets(cfg):
    missing = []
    for role in ("student", "teacher", "reference"):
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
    with console_log(run/"console.log"):
        with Progress(run,"campaign",9,unit="stage",eta=False) as progress:
            atomic_json(run/'campaign.json',{'status':'running','pid':os.getpid(),'attempt_id':progress.attempt_id,
                'started_at':progress.started_at,'seed':42,'parameter_updates':0,'automatic_opd':False})
            return _execute(cfg,progress)


def _execute(cfg,progress):
    run = project_path(cfg["run_dir"])
    engine_python = str(ROOT / cfg["engine_python"]) if cfg.get("engine_python") else sys.executable
    def gpu(name,count,command,*,interpreter=None):
        from run_on_idle_gpus import tee_process,process_ticks
        expected=[interpreter or engine_python,*command]
        progress.phase=name;progress.event("stage_requested",name=name,command=expected)
        env=child_environment(expected[0]);env['STEER_OPD_RUN_ROOT']=str(run)
        # Adopt a live stage after parent interruption instead of admitting it twice.
        for previous in sorted((run/'attempts').glob(f'{name}_*'),reverse=True):
            queue_path=previous/'queue.json'
            if not queue_path.exists():continue
            queue=read_json(queue_path)
            if queue.get('command')!=expected or process_ticks(queue['pid'])!=queue.get('start_ticks') or queue.get('start_ticks') is None:continue
            progress.event('adopting_live_stage',name=name,attempt=str(previous),pid=queue['pid'])
            while process_ticks(queue['pid'])==queue['start_ticks']:time.sleep(1)
            marker=previous/'process.json'
            if not marker.exists() or read_json(marker).get('exit_code')!=0: raise RuntimeError(f'Adopted stage did not finish successfully: {previous}')
            progress.update(key=name);return
        stage_output=(run/'references'/name.removeprefix('reference_') if name.startswith('reference_') else
                      run/'native_scores'/name.removeprefix('native_') if name.startswith('native_') else
                      run/'select' if name=='eight_way_select' else run/'confirm' if name=='frozen_confirm' else run/name)
        if stage_output.exists():
            progress.event('cache_validation_start',name=name)
            trial_env={**env,'STEER_OPD_CACHE_ONLY':'1','CUDA_VISIBLE_DEVICES':''}
            with (run/f'{name}_cache_validation.log').open('ab') as log:
                trial=subprocess.Popen(expected,cwd=ROOT,env=trial_env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
                code=tee_process(trial,log)
            if code==0:
                progress.update(key=name);return
            if code!=75:raise subprocess.CalledProcessError(code,expected)
        attempt=run/'attempts'/f'{name}_{time.time_ns()}'
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
        progress.update(key=name)
    codecs = load_codecs(cfg)
    questions = prepare_initial_jobs(cfg, codecs)
    for role in ("teacher", "reference", "student"):
        job_path = run / "jobs" / f"reference_{role}.json"
        frozen_json(job_path, generation_job(cfg, codecs[role], role, questions))
        gpu(f"reference_{role}", cfg["teacher_gpus"] if role == "teacher" else 1,
            ["scripts/generate_static_sources.py", "--job", str(job_path)])
    bank_cfg = {**cfg, "questions": f"{cfg['data_dir']}/questions.json", "bank_dir": f"{cfg['run_dir']}/banks",
                "reference_dirs": {r: f"{cfg['run_dir']}/references/{r}" for r in ("teacher", "reference", "student")},
                "minimum_pairs": 2*cfg["minimum_split_questions"]}
    frozen_json(run / "jobs/banks.json", bank_cfg)
    gpu("banks", 1, ["scripts/prepare_compass_banks.py", "--config", str(run / "jobs/banks.json")])
    common = read_json(run / "banks/questions.json")
    source_job = generation_job(cfg, codecs["student"], "student", common, source=True)
    frozen_json(run / "jobs/source.json", source_job)
    gpu("source", 1, ["scripts/generate_static_sources.py", "--job", str(run / "jobs/source.json")])
    rows = [read_json(run / "source" / f"{r['key']}.json") for r in source_job["requests"]]
    frozen_json(run / "source_rows.json", rows)
    base_job = make_base_job(cfg, codecs, f"{cfg['run_dir']}/source_rows.json")
    frozen_json(run / "jobs/alignment_and_scoring.json", base_job)
    for role in ("teacher", "reference"):
        gpu(f"native_{role}", cfg["teacher_gpus"] if role == "teacher" else 1,
            ["scripts/score_compass_sources.py", "--job", str(run / "jobs/alignment_and_scoring.json"), "--role", role, "--execute"])
    select_job, confirm_job = prepare_screen_jobs(cfg, base_job, codecs)
    screen_python = str(ROOT / cfg["screen_python"]) if cfg.get("screen_python") else engine_python
    gpu("eight_way_select", 1, ["scripts/run_static_overlap_screen.py", "--job", str(run / "jobs/select.json"), "--execute"], interpreter=screen_python)
    summaries = read_json(run / "select/summaries.json")
    ranked = sorted(summaries, key=lambda k: (-summaries[k]["question_weighted_delta_pp"], k))
    chosen = ranked[:cfg["confirm_top_n"]]
    frozen_json(run / "selected_schemes.json", {"ids": chosen, "selection_uses_confirm": False, "parameter_updates": 0,
                "baseline_is_best_on_select": summaries[ranked[0]]["question_weighted_delta_pp"] <= 0})
    confirm_job["conditions"] = [c for c in confirm_job["conditions"] if c["id"] in chosen]
    frozen_json(run / "jobs/confirm.json", confirm_job)
    gpu("frozen_confirm", 1, ["scripts/run_static_overlap_screen.py", "--job", str(run / "jobs/confirm.json"), "--execute"], interpreter=screen_python)
    with progress.phase_scope("archive_training_reuse_artifacts"):
        export_artifacts(cfg,run)
    atomic_json(run / "campaign.json", {"status": "awaiting_user_review", "parameter_updates": 0,
                "select_schemes": len(select_job['conditions']), "confirm_ids": chosen, "automatic_opd": False,
                "recorded_metrics": ["overlap", "P_overlap"], "gate_metric": cfg.get("gate_metric", "overlap")})


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
    frozen_json(run / "config.json", cfg)
    missing = missing_assets(cfg)
    codecs = load_codecs(cfg)
    questions = prepare_initial_jobs(cfg, codecs)
    policy = execution_policy(cfg)
    held = bool(cfg.get("execution_hold"))
    status = {**policy, "status": "awaiting_user_approval" if held else
              ("prerequisites_pending" if cfg.get("execution_blockers") else
               ("missing_assets" if missing else "ready_for_execution")), "config_hash": digest(cfg),
              "asset_status": "missing_assets" if missing else "present_not_gpu_validated",
              "screen_backend": cfg["screen_backend"],
              "recorded_metrics": ["overlap", "P_overlap"], "gate_metric": cfg.get("gate_metric", "overlap"),
              "missing_assets": missing, "parameter_updates": 0, "automatic_opd": False,
              "training_questions_prepared": len(questions), "nonzero_scheme_count": len(make_base_job(cfg,codecs,'unused')['conditions']),
              "native_supports": {r: {"mapped_tokens": len(c.valid_ids), "support_size": c.support_size,
                                      "mapping_sha256": c.mapping_sha256} for r, c in codecs.items()},
              "stages": ["T_Ref_S_texts", "common_S0_banks", "unsteered_source_top16", "native_T_Ref_span_scores",
                         "select_threshold", "all_eight_fixed_mask_scores", "frozen_confirm", "user_review"]}
    atomic_json(run / "preflight.json", status)
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
