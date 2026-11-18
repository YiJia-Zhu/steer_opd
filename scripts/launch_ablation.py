#!/usr/bin/env python3
"""Launch frozen-prefix ablation or parameter overlap experiments; zero OPD updates."""
import argparse
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from steer_opd.compass_campaign import frozen_json, scheme_conditions, validate_config
from steer_opd.compass_job import model_path, MODEL_ROLES
from steer_opd.method_config import TEACHER_DIRECTION_VERSION
from steer_opd.experiment_runtime import child_environment, console_log, source_revision, timestamp
from steer_opd.resumable_core import ROOT, atomic_json, digest, read_json
from run_ablation import missing_assets
from run_on_idle_gpus import parse_gpu_ids,tee_process
from steer_opd.stage_io import configure_layout,metadata_dir,logs_dir,component_target,frozen_data_dir
from steer_opd.artifact_reuse import StageArtifacts,contract
from steer_opd.tensor_checkpoint import DEFAULT_CANDIDATE_FORMAT,SUPPORTED_FORMATS
from steer_opd.generation_policy import DEFAULT_MAX_GENERATION_LENGTH,DEFAULT_PRESENCE_PENALTY,presence_penalty


def run_command(command,env):
    """Forward native progress live, including into the startup console tee."""
    child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
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
    p.add_argument('--run-name',choices=['ablation','param','compar'],default='ablation')
    p.add_argument('--param-name',default=None)
    p.add_argument('--question-count',type=int,default=1000)
    p.add_argument('--input-data-dir',default=frozen_data_dir())
    p.add_argument('--base-question-source',default='',help='Optional read-only frozen 100-question source')
    p.add_argument('--data-policy',choices=['select_confirm','all_train'],default='select_confirm',
        help='all_train evaluates the whole training pool once, without selection or confirmation')
    p.add_argument('--teacher-ablation',type=int,choices=[0,1],default=0,
        help='Add teacher-space S-T schemes 5..8; no GPU work unless --run')
    p.add_argument('--student-baseline-run',default='',help='Reuse a committed student-arm baseline for teacher-only supplementation')
    p.add_argument('--teacher-layer',type=int,default=29,help='0-based 29 = 30th of 40 teacher blocks (75%% depth)')
    p.add_argument('--stop-after',choices=['source_overlap','panel'],default='panel',
        help='Stop after unsteered source overlap plots; panel requires explicit selection')
    p.add_argument('--strength-policies',choices=['fixed'],default='fixed',help='Overlap uses fixed strength only; annealing belongs to future training')
    p.add_argument('--phase',choices=['overlap'],default='overlap',help='Frozen-prefix overlap only; OPD and method baselines are not implemented')
    p.add_argument('--operator',choices=['slerp','additive'],default='slerp')
    p.add_argument('--alpha',type=float,default=.1);p.add_argument('--layer',type=int,default=26)
    p.add_argument('--gate-metric',choices=['overlap','P_overlap'],default='overlap')
    p.add_argument('--rank-metric',choices=['overlap','P_overlap'],default='overlap')
    p.add_argument('--gate-target-rate',type=float,default=.2)
    p.add_argument('--teacher-tp',type=int,choices=[1,2],default=2)
    p.add_argument('--gpu-memory',type=float,default=.75);p.add_argument('--wait-seconds',type=int,default=3600)
    p.add_argument('--reference-max-tokens',type=int,default=DEFAULT_MAX_GENERATION_LENGTH);p.add_argument('--source-max-tokens',type=int,default=DEFAULT_MAX_GENERATION_LENGTH)
    p.add_argument('--sv-max-context-len',type=int,default=2048,help='Prompt+answer limit for SV-training activation extraction')
    p.add_argument('--reference-temperature',type=float,default=1.);p.add_argument('--reference-top-p',type=float,default=1.)
    p.add_argument('--presence-penalty',type=float,default=DEFAULT_PRESENCE_PENALTY,
        help='Teacher and student generation penalty; fixed-prefix scores remain raw')
    p.add_argument('--candidate-tensor-format','--candidate-tensor-compression',dest='candidate_tensor_compression',
        choices=[*SUPPORTED_FORMATS,'none','gzip'],default=DEFAULT_CANDIDATE_FORMAT,
        help='New candidate storage; old none/gzip options remain supported')
    p.add_argument('--confirm-top-n',type=int,default=None)
    p.add_argument('--hf-parity',type=int,choices=[0,1],default=0,help='Optional HF numerical comparison; default uses EasySteer only')
    return p


def build_config(args):
    import re
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*',args.run_name):
        raise ValueError('RUN_NAME must contain only letters, digits, underscores or hyphens')
    gpu_ids=parse_gpu_ids(args.gpu_ids)
    if len(gpu_ids)<args.teacher_tp:raise ValueError('GPU_IDS must contain at least TEACHER_TP cards')
    policies=args.strength_policies.split(',')
    if policies!=['fixed']:raise ValueError('Overlap uses four fixed-strength schemes; annealing is deferred to training')
    count=(8 if args.teacher_ablation else 4)*len(policies)
    if args.schemes=='all':ids=[f'scheme_{i:02d}' for i in range(1,count+1)]
    else:
        try:ids=[f'scheme_{int(i.strip().removeprefix("scheme_")):02d}' for i in args.schemes.split(',')]
        except ValueError:raise ValueError(f'SCHEME_IDS must be all or IDs 1..{count} separated by commas') from None
    cfg=read_json(ROOT/'configs/ablation.json')
    cfg.update(id=f'{args.run_name}_{args.phase}',experiment=args.run_name,phase=args.phase,
        run_name=args.run_name,param_name=args.param_name,question_count=args.question_count,input_data_dir=args.input_data_dir,
        scheme_ids=ids,operator=args.operator,alpha=args.alpha,layers=[args.layer],
        stop_after=args.stop_after,
        strength_policies=policies,
        gate_metric=args.gate_metric,rank_metric=args.rank_metric,gate_target_rate=args.gate_target_rate,
        teacher_gpus=args.teacher_tp,gpu_memory_utilization=args.gpu_memory,gpu_wait_seconds=args.wait_seconds,
        reference_max_tokens=args.reference_max_tokens,source_max_tokens=args.source_max_tokens,
        sv_max_context_len=args.sv_max_context_len,
        reference_temperature=args.reference_temperature,reference_top_p=args.reference_top_p,
        presence_penalty=presence_penalty({'presence_penalty':args.presence_penalty}),
        confirm_top_n=min(args.confirm_top_n if args.confirm_top_n is not None else count,len(ids)),
        include_teacher_ablation=bool(args.teacher_ablation),teacher_layers=[args.teacher_layer],
        teacher_direction_version=TEACHER_DIRECTION_VERSION,
        startup_hf_parity=bool(args.hf_parity))
    cfg['candidate_tensor_compression']=args.candidate_tensor_compression
    if args.base_question_source:
        cfg['base_question_source']=args.base_question_source
    if args.data_policy=='all_train':
        cfg['data_policy']='all_train'
        cfg['training_input_data_dir']=args.input_data_dir
        for key in ('select_fraction','minimum_split_questions','confirm_top_n'):
            cfg.pop(key,None)
    if args.student_baseline_run:
        if not args.teacher_ablation or any(int(s.removeprefix('scheme_'))<5 for s in ids):
            raise ValueError('An external student baseline is only for teacher-only schemes 5..8')
        cfg['student_baseline_run']=args.student_baseline_run
    cfg=configure_layout(cfg)
    cfg['id']=f"{args.run_name}_{args.phase}_{cfg['param_name']}"
    if (not 0<args.gpu_memory<1 or args.wait_seconds<0 or args.reference_max_tokens<1 or args.source_max_tokens<1
            or not math.isfinite(args.reference_temperature) or args.reference_temperature<0 or not 0<args.reference_top_p<=1):
        raise ValueError('Invalid memory fraction, wait, generation length or sampling parameters')
    # The original held template is unchanged. Only manual --run creates an executable run.
    cfg.update(execution_hold=None,execution_blockers=[],
        gpu_acceptance_receipt=str(component_target(cfg,'acceptance')/'passed.json'))
    validate_config(cfg)
    return cfg


def run_validation(cfg,gpu_ids,config_path):
    from steer_opd.checkpoint_identity import checkpoint_content_id
    from steer_opd.experiment_runtime import Progress
    output=ROOT/component_target(cfg,'acceptance')
    backends=('hf','easysteer') if cfg.get('startup_hf_parity') else ('easysteer',)
    with Progress(output/'progress','startup_validation',len(MODEL_ROLES)*len(backends),unit='stage',eta=False,show_bar=False) as progress:
        with progress.phase_scope('checkpoint_content_hashes'):
            identities={role:checkpoint_content_id(model_path(cfg[role]),receipt_dir=output/'checkpoints')
                        for role in MODEL_ROLES}
        binding=digest({'source_revision':source_revision()[0],'checkpoints':identities,
                        'student':cfg['student'],'layers':cfg['layers'],'alpha':cfg['alpha'],
                        'operator':cfg.get('operator','slerp'),
                        'presence_penalty':presence_penalty(cfg),
                        'additive_scaling':'raw_mean_difference_v1' if cfg.get('operator')=='additive' else None,
                        'teacher_gpus':cfg['teacher_gpus'],'hf_parity':cfg.get('startup_hf_parity',False),
                        'direction_version':cfg['direction_version'],
                        'sv_max_context_len':cfg['sv_max_context_len'],
                        'include_teacher_ablation':cfg.get('include_teacher_ablation',False),
                        'teacher_layers':cfg.get('teacher_layers'),
                        'teacher_direction_version':cfg.get('teacher_direction_version'),
                        'experiment':cfg['experiment'],'phase':cfg['phase'],
                        'strength_policies':cfg['strength_policies'],
                        'stop_after':cfg['stop_after'],
                        'roles':list(MODEL_ROLES),'protocol':'ablation_overlap_gpu_validation_v5_top16_raw'})
        cache=StageArtifacts(cfg)
        science=contract('acceptance',{'validation_binding':binding})
        found=cache.find('acceptance',science,output)
        if found:
            owner,packet=cache.bind(output,found,science)
            passed=read_json(owner/'passed.json')
            if passed['binding']!=binding:raise ValueError('Acceptance receipt identity changed')
            executable={**cfg,'gpu_acceptance_binding':binding,'gpu_acceptance_receipt':str(owner/'passed.json')}
            frozen_json(config_path,executable)
            return executable
        job={**cfg,'stage':'static_gpu_acceptance','gpu_acceptance_binding':binding}
        job.pop('gpu_acceptance_receipt')
        job_path=output/'job.json';frozen_json(job_path,job)
        python=str(ROOT/cfg['engine_python'])
        for role in MODEL_ROLES:
            for backend in backends:
                progress.phase=f'{role}_{backend}'
                progress.event('stage_requested',name=progress.phase)
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
            'roles':list(MODEL_ROLES),'parameter_updates':0,
            'reports':{r:read_json(output/r/'easysteer/report.json') for r in MODEL_ROLES}})
    cache.publish('acceptance',science,output,files=[output/'passed.json',output/'job.json',
        *(f for r in MODEL_ROLES for b in backends for f in (output/r/b).rglob('*')
          if f.is_file() and f.name not in ('progress.json','events.jsonl') and 'attempts' not in f.parts)])
    executable={**cfg,'gpu_acceptance_binding':binding}
    frozen_json(config_path,executable)
    return executable


def main():
    args=parser().parse_args();cfg=build_config(args);run=ROOT/cfg['run_dir']
    meta=ROOT/metadata_dir(cfg);logs=ROOT/logs_dir(cfg)
    config_path=meta/'launch_config.json'
    if args.status:
        status_config=next((p for p in (config_path,meta/'launch_request.json',meta/'layout_config.json') if p.exists()),meta/'layout_config.json')
        if status_config.exists():subprocess.run([sys.executable,'scripts/show_ablation_progress.py','--config',str(status_config)],check=True)
        else:print(f'No execution yet: {run}')
        return
    print(f"RUN_NAME={cfg['run_name']}; PARAM_NAME={cfg['param_name']}\nRun: {run}\nGPU pool (physical IDs): {args.gpu_ids}; teacher TP={cfg['teacher_gpus']}",flush=True)
    print(f"Experiment: {cfg['experiment']} / {cfg['phase']}; frozen prefixes; optimizer updates=0",flush=True)
    if cfg.get('data_policy')=='all_train':
        print(f"Questions: {cfg['question_count']}; one training pool for SV, overlap and OPD; data={cfg['data_dir']}",flush=True)
    else:
        print(f"Questions: {cfg['question_count']}; select fraction={cfg['select_fraction']}; confirm schemes={cfg['confirm_top_n']}; data={cfg['data_dir']}",flush=True)
    print(f"Stop after: {cfg['stop_after']}; source_overlap skips teacher answer generation, SV capture/banks/injection and OPD",flush=True)
    print(f"Operator={cfg.get('operator','slerp')}, strength={cfg['alpha']}, layer={cfg['layers']}, seed=42; gate={cfg['gate_metric']}, rank={cfg['rank_metric']}",flush=True)
    print(f"SV extraction: all samples, mean over retained answer tokens; prompt+answer limit={cfg['sv_max_context_len']}; no finish/correctness/repetition filter",flush=True)
    roles={c['steer_role'] for c in scheme_conditions(cfg)}
    print('Direction: S-T in teacher space; four schemes with a frozen student baseline' if roles=={'teacher'}
          else 'Directions: selected student T-S / optional teacher S-T schemes; shared no-SV baseline',flush=True)
    if cfg.get('include_teacher_ablation'):
        print(f"Teacher ablation: S-T in teacher space, layer={cfg['teacher_layers']}; schemes 5..8; student stays unsteered",flush=True)
    print('Annealing is deferred to actual training batches; no question-order annealing in overlap',flush=True)
    print('Metrics v4: Top16 prefix-compatible count + raw Top16 probability mass matching; no tail renormalization; old metric receipts cannot be reused',flush=True)
    print('Startup validation: '+('HF + EasySteer comparison' if cfg['startup_hf_parity'] else 'EasySteer only (HF model forward disabled)'),flush=True)
    if cfg['stop_after']=='panel':
        for c in scheme_conditions(cfg):print(f"  {c['id']}: {c['direction']} / {c['gate']} / {c['granularity']} / {c['strength_policy']}")
    for step,folder in cfg['step_dirs'].items():print(f'Step {step}: {folder}',flush=True)
    print('Reuse: SHA-verified upstream artifacts; changing alpha does not invalidate rollouts or SV banks.',flush=True)
    missing=missing_assets(cfg)
    print(f'Weights: {missing or "present; content hashes checked before launch"}',flush=True)
    print('Resume: complete stages/candidates only. Both overlap metrics are recorded; optimizer updates=0.',flush=True)
    # A preview stays read-only even when the existing run needs migration.
    frozen=meta/'launch_request.json'
    conflict=None
    if (meta/'historical_run.json').exists() and not frozen.exists():
        conflict='This PARAM_NAME contains a completed historical panel; use a new PARAM_NAME.'
    elif meta.exists() and (meta/'config.json').exists() and not frozen.exists():
        conflict='This PARAM_NAME already contains another run.'
    elif frozen.exists() and read_json(frozen)!=cfg:
        conflict='This PARAM_NAME already has different parameters.'
    if not args.run:
        if conflict:
            print(conflict+' Existing artifacts preserved; execution requires an explicit identity migration or a separate run.',flush=True)
        print('CPU preview only. Use bash scripts/run_ablation.sh --run to explicitly launch.')
        return
    # Refuse changed identities before any validation/model or artifact writes.
    if conflict:raise ValueError(conflict+' Do not mix existing artifacts with this configuration.')
    if missing:raise FileNotFoundError('Weights incomplete; finish your manual download before --run')
    def interrupted(signum,frame):raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    run.mkdir(parents=True,exist_ok=True)
    frozen_json(frozen,cfg)
    atomic_json(meta/'launches'/f'{time.time_ns()}.json',{'started_at':timestamp(),'gpu_ids':args.gpu_ids,
        'argv':sys.argv,'shell_source':(ROOT/'scripts/run_ablation.sh').read_text()})
    with console_log(logs/'startup.log'):
        # Also validates installed public APIs/fingerprints; no model is loaded by this command.
        env=child_environment(str(ROOT/cfg['engine_python']))
        run_command([str(ROOT/cfg['engine_python']),'scripts/check_easysteer_runtime.py'],env)
        run_validation(cfg,args.gpu_ids,config_path)
    env={**os.environ,'STEER_OPD_GPU_IDS':args.gpu_ids}
    run_command([sys.executable,'scripts/run_ablation.py','--config',str(config_path),'--execute','--resume'],env)


if __name__=='__main__':main()
