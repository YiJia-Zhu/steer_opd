#!/usr/bin/env python3
"""One manual launch command: CPU preview, cached GPU validation, then the panel."""
import argparse
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from steer_opd.compass_campaign import frozen_json, scheme_conditions, validate_config
from steer_opd.compass_job import model_path
from steer_opd.experiment_runtime import child_environment, console_log, source_revision, timestamp
from steer_opd.resumable_core import ROOT, atomic_json, digest, read_json
from run_compass_static_panel import missing_assets
from run_on_idle_gpus import parse_gpu_ids,tee_process


def run_command(command,env):
    """Forward native progress live, including into the startup console tee."""
    child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    try:code=tee_process(child,None)
    except BaseException:
        if child.poll() is None:
            child.terminate()
            try:child.wait(timeout=40)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        raise
    if code:raise subprocess.CalledProcessError(code,command)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    modes=p.add_mutually_exclusive_group()
    modes.add_argument('--check',action='store_true',help='CPU preview only; never query or load GPUs')
    modes.add_argument('--run',action='store_true',help='Explicitly start/resume validation and selected schemes')
    modes.add_argument('--status',action='store_true',help='Read saved progress for this run')
    p.add_argument('--gpu-ids',default='0,1');p.add_argument('--schemes',default='all')
    p.add_argument('--run-name',default='static_panel_v1')
    p.add_argument('--alpha',type=float,default=.1);p.add_argument('--layer',type=int,default=26)
    p.add_argument('--gate-metric',choices=['overlap','P_overlap'],default='overlap')
    p.add_argument('--rank-metric',choices=['overlap','P_overlap'],default='overlap')
    p.add_argument('--gate-target-rate',type=float,default=.2)
    p.add_argument('--teacher-tp',type=int,choices=[1,2],default=2)
    p.add_argument('--gpu-memory',type=float,default=.75);p.add_argument('--wait-seconds',type=int,default=3600)
    p.add_argument('--reference-max-tokens',type=int,default=12288);p.add_argument('--source-max-tokens',type=int,default=7168)
    p.add_argument('--reference-temperature',type=float,default=.7);p.add_argument('--reference-top-p',type=float,default=.95)
    p.add_argument('--confirm-top-n',type=int,default=2)
    return p


def build_config(args):
    import re
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*',args.run_name):
        raise ValueError('RUN_NAME must contain only letters, digits, underscores or hyphens')
    gpu_ids=parse_gpu_ids(args.gpu_ids)
    if len(gpu_ids)<args.teacher_tp:raise ValueError('GPU_IDS must contain at least TEACHER_TP cards')
    if args.schemes=='all':ids=[f'scheme_{i:02d}' for i in range(1,9)]
    else:
        try:ids=[f'scheme_{int(i.strip().removeprefix("scheme_")):02d}' for i in args.schemes.split(',')]
        except ValueError:raise ValueError('SCHEME_IDS must be all or IDs 1..8 separated by commas') from None
    cfg=read_json(ROOT/'configs/static_dual_overlap.json')
    cfg.update(id=args.run_name,run_dir=f'runs/{args.run_name}',scheme_ids=ids,alpha=args.alpha,layers=[args.layer],
        gate_metric=args.gate_metric,rank_metric=args.rank_metric,gate_target_rate=args.gate_target_rate,
        teacher_gpus=args.teacher_tp,gpu_memory_utilization=args.gpu_memory,gpu_wait_seconds=args.wait_seconds,
        reference_max_tokens=args.reference_max_tokens,source_max_tokens=args.source_max_tokens,
        reference_temperature=args.reference_temperature,reference_top_p=args.reference_top_p,
        confirm_top_n=min(args.confirm_top_n,len(ids)))
    if (not 0<args.gpu_memory<1 or args.wait_seconds<0 or args.reference_max_tokens<1 or args.source_max_tokens<1
            or not math.isfinite(args.reference_temperature) or args.reference_temperature<0 or not 0<args.reference_top_p<=1):
        raise ValueError('Invalid memory fraction, wait, generation length or sampling parameters')
    # The original held template is unchanged. Only manual --run creates an executable run.
    cfg.update(execution_hold=None,execution_blockers=[],
        gpu_acceptance_receipt=f'runs/{args.run_name}/acceptance/passed.json')
    validate_config(cfg)
    return cfg


def run_validation(cfg,gpu_ids,config_path):
    from steer_opd.checkpoint_identity import checkpoint_content_id
    from steer_opd.experiment_runtime import Progress
    output=ROOT/cfg['run_dir']/'acceptance'
    with Progress(output/'progress','startup_validation',6,unit='stage',eta=False) as progress:
        with progress.phase_scope('checkpoint_content_hashes'):
            identities={role:checkpoint_content_id(model_path(cfg[role]),receipt_dir=output/'checkpoints')
                        for role in ('student','reference','teacher')}
        binding=digest({'source_revision':source_revision()[0],'checkpoints':identities,
                        'student':cfg['student'],'layers':cfg['layers'],'alpha':cfg['alpha'],
                        'teacher_gpus':cfg['teacher_gpus'],'protocol':'static_gpu_validation_v1'})
        job={**cfg,'stage':'static_gpu_acceptance','gpu_acceptance_binding':binding}
        job.pop('gpu_acceptance_receipt')
        job_path=output/'job.json';frozen_json(job_path,job)
        python=str(ROOT/cfg['engine_python'])
        for role in ('student','reference','teacher'):
            for backend in ('hf','easysteer'):
                progress.phase=f'{role}_{backend}'
                result=output/role/backend/'complete.json'
                command=[python,'scripts/validate_static_gpu.py','--job',str(job_path),'--role',role,'--backend',backend]
                env=child_environment(python)
                if result.exists():
                    # Validate the cached manifest and binding without admitting a GPU.
                    run_command(command,{**env,'CUDA_VISIBLE_DEVICES':'','STEER_OPD_CACHE_ONLY':'1'})
                else:
                    attempt=output/'attempts'/f'{role}_{backend}_{time.time_ns()}'
                    launch=[sys.executable,'scripts/run_on_idle_gpus.py','--count',str(cfg['teacher_gpus'] if role=='teacher' else 1),
                        '--gpu-ids',gpu_ids,'--wait-seconds',str(cfg['gpu_wait_seconds']),'--run-dir',str(attempt),'--',*command]
                    run_command(launch,env)
                progress.update(key=f'{role}_{backend}')
        atomic_json(output/'passed.json',{'status':'passed','binding':binding,'completed_at':timestamp(),
            'roles':['student','reference','teacher'],'parameter_updates':0,
            'reports':{r:read_json(output/r/'easysteer/report.json') for r in ('student','reference','teacher')}})
    frozen_json(config_path,{**cfg,'gpu_acceptance_binding':binding})


def main():
    args=parser().parse_args();cfg=build_config(args);run=ROOT/cfg['run_dir']
    config_path=run/'launch_config.json'
    if args.status:
        status_config=config_path if config_path.exists() else run/'launch_request.json'
        if status_config.exists():subprocess.run([sys.executable,'scripts/show_static_progress.py','--config',str(status_config)],check=True)
        else:print(f'No execution yet: {run}')
        return
    print(f"Run: {run}\nGPU pool (physical IDs): {args.gpu_ids}; teacher TP={cfg['teacher_gpus']}",flush=True)
    print(f"Slerp={cfg['alpha']}, layer={cfg['layers']}, seed=42; gate={cfg['gate_metric']}, rank={cfg['rank_metric']}",flush=True)
    for c in scheme_conditions(cfg):print(f"  {c['id']}: {c['direction']} / {c['gate']} / {c['granularity']}")
    missing=missing_assets(cfg)
    print(f'Weights: {missing or "present; content hashes checked before launch"}',flush=True)
    print('Resume: complete stages/candidates only. Both overlap metrics are recorded; optimizer updates=0.',flush=True)
    # Refuse changes before running any validation/model or rewriting artifacts.
    frozen=run/'launch_request.json'
    if run.exists() and any(run.iterdir()) and not frozen.exists():
        raise ValueError('This RUN_NAME already contains another run. Choose a new RUN_NAME.')
    if frozen.exists() and read_json(frozen)!=cfg:
        raise ValueError('This RUN_NAME already has different parameters. Choose a new RUN_NAME.')
    if not args.run:
        print('CPU preview only. Use bash scripts/run_static_panel.sh --run to explicitly launch.')
        return
    if missing:raise FileNotFoundError('Weights incomplete; finish your manual download before --run')
    def interrupted(signum,frame):raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    run.mkdir(parents=True,exist_ok=True)
    frozen_json(frozen,cfg)
    atomic_json(run/'launches'/f'{time.time_ns()}.json',{'started_at':timestamp(),'gpu_ids':args.gpu_ids,
        'argv':sys.argv,'shell_source':(ROOT/'scripts/run_static_panel.sh').read_text()})
    with console_log(run/'startup.log'):
        # Also validates installed public APIs/fingerprints; no model is loaded by this command.
        env=child_environment(str(ROOT/cfg['engine_python']))
        run_command([str(ROOT/cfg['engine_python']),'scripts/check_easysteer_runtime.py'],env)
        run_validation(cfg,args.gpu_ids,config_path)
    env={**os.environ,'STEER_OPD_GPU_IDS':args.gpu_ids}
    run_command([sys.executable,'scripts/run_compass_static_panel.py','--config',str(config_path),'--execute','--resume'],env)


if __name__=='__main__':main()
