"""Prepare/check a VERL recipe; training requires --run and a matching GPU receipt."""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import urllib.request

from steer_opd.resumable_core import ROOT, atomic_json, digest, read_json
from steer_opd.training.config import TrainingLossConfig
from steer_opd.training.rollout_logprobs import SOURCE as STUDENT_LOGPROB_SOURCE
from steer_opd.training.verl_plan import (VERL_ARCHIVE_SHA256,VERL_REVISION,VERL_SOURCE,
                                         VerlSettings,build_verl_plan,gpu_ids,verl_overrides)


def bootstrap_source():
    parent=ROOT/'.runtime/verl_training';parent.mkdir(parents=True,exist_ok=True)
    archive=parent/(VERL_REVISION+'.tar.gz')
    if not archive.exists():
        with urllib.request.urlopen('https://codeload.github.com/volcengine/verl/tar.gz/'+VERL_REVISION,timeout=60) as response:
            data=response.read()
        if hashlib.sha256(data).hexdigest()!=VERL_ARCHIVE_SHA256:raise ValueError('VERL source archive checksum changed')
        archive.write_bytes(data)
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=VERL_ARCHIVE_SHA256:raise ValueError('Pinned VERL archive changed')
    target=ROOT/VERL_SOURCE
    if not target.exists():
        with tarfile.open(archive) as tar:tar.extractall(parent,filter='data')
    return target


def activate_verl():
    source=ROOT/VERL_SOURCE
    if not source.is_dir():raise FileNotFoundError('Prepare pinned VERL source with --bootstrap-source')
    archive=ROOT/'.runtime/verl_training'/(VERL_REVISION+'.tar.gz')
    if not archive.exists() or hashlib.sha256(archive.read_bytes()).hexdigest()!=VERL_ARCHIVE_SHA256:
        raise ValueError('Pinned VERL archive is missing or changed')
    # Verify actual extracted source, not only the archive's label.
    with tarfile.open(archive) as tar:
        for item in tar.getmembers():
            if item.isfile() and (item.name.endswith(('.py','.yaml')) or item.name.endswith('/version/version')):
                expected=tar.extractfile(item).read()
                actual=source.parent/item.name
                if not actual.is_file() or actual.read_bytes()!=expected:raise ValueError(f'Pinned VERL source changed: {item.name}')
    sys.path.insert(0,str(source))
    os.environ['PYTHONPATH']=os.pathsep.join([str(source),str(ROOT/'src'),os.environ.get('PYTHONPATH','')])
    return source


def compose_verl(plan):
    activate_verl()
    from hydra import compose,initialize_config_dir
    from omegaconf import OmegaConf
    with initialize_config_dir(config_dir=str(ROOT/VERL_SOURCE/'verl/trainer/config'),version_base=None):
        config=compose(config_name='ppo_trainer',overrides=['model_engine=dp'])
    for key,value in verl_overrides(plan).items():OmegaConf.update(config,key,value,merge=False,force_add=True)
    OmegaConf.update(config,'steer_opd',plan,merge=False,force_add=True)
    # Canonical runtime configs are materialized by Hydra; force resolved values
    # before any Ray workers start so a missing/renamed upstream field fails here.
    OmegaConf.resolve(config)
    from verl.utils.config import validate_config
    validate_config(config=config,use_reference_policy=False,use_critic=False)
    return config


def prepare_data(plan):
    import pandas as pd
    from steer_opd.math_data import TASK_SUFFIX
    root=ROOT/plan['output'];source=ROOT/plan['input_data_dir']
    questions={str(q['id']):q for q in read_json(source/'questions_source.json')}
    if (len(questions)!=plan['train_questions'] or digest(list(questions))!=plan['training_ids_sha256']
            or plan.get('data_policy')!='all_train'):
        raise ValueError('Training parquet must contain the complete frozen question pool')
    for name,ids in [('train.parquet',list(questions))]:
        records=[]
        for qid in ids:
            row=questions[str(qid)]
            records.append({'data_source':'steer_opd','prompt':[{'role':'user','content':row['question']+TASK_SUFFIX}],
                'ability':'math','reward_model':{'style':'rule','ground_truth':''},
                'extra_info':{'question_id':str(qid),'question':row['question']}})
        output=root/'data'/name
        output.parent.mkdir(parents=True,exist_ok=True)
        if output.exists():
            old=pd.read_parquet(output).to_json(orient='records')
            if json.loads(old)!=json.loads(pd.DataFrame(records).to_json(orient='records')):
                raise ValueError('Frozen training dataset changed')
        else:pd.DataFrame(records).to_parquet(output,index=False)
    files={name:hashlib.sha256((root/'data'/name).read_bytes()).hexdigest()
           for name in ('train.parquet',)}
    if plan.get('evaluation'):
        from steer_opd.training.benchmarks import benchmark_rows
        records,evaluation=benchmark_rows(list(questions.values()))
        if evaluation!=plan['evaluation']:raise ValueError('Frozen benchmark identities changed')
        path=root/'data/benchmarks.parquet';frame=pd.DataFrame(records)
        if path.exists():
            if json.loads(pd.read_parquet(path).to_json(orient='records'))!=json.loads(frame.to_json(orient='records')):
                raise ValueError('Frozen benchmark parquet changed')
        else:frame.to_parquet(path,index=False)
        files[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    manifest={'source_sha256':plan['input_sha256'],'files':files,'train_questions':plan['train_questions'],
              'data_policy':'all_train','training_ids_sha256':plan['training_ids_sha256']}
    dest=root/'metadata/data_manifest.json'
    if dest.exists() and read_json(dest)!=manifest:raise ValueError('Frozen training data manifest changed')
    atomic_json(dest,manifest)
    return root


def freeze_prepare(plan):
    root=ROOT/plan['output'];path=root/'metadata/plan.json'
    if path.exists() and read_json(path)!=plan:raise ValueError('Use a new training run name; prepared plan is immutable')
    prepare_data(plan)
    atomic_json(path,plan)
    config=compose_verl(plan)
    from omegaconf import OmegaConf
    atomic_json(root/'metadata/verl_config.json',OmegaConf.to_container(config,resolve=True))
    return path


def validate_plan(plan):
    if plan['binding']!=digest({k:v for k,v in plan.items() if k!='binding'}):
        raise ValueError('Frozen training plan content identity changed')
    for name,expected in plan['input_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Frozen source questions or split changed')
    for name,expected in plan['implementation_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Training implementation changed: {name}; prepare a new run identity')


def validate_receipt(plan,path):
    validate_plan(plan)
    if not path:raise RuntimeError('Training GPU acceptance is pending; --run requires --acceptance-receipt')
    receipt=read_json(Path(path))
    roles=['student_fsdp_rollout','teacher_scoring']
    if plan['training_loss']['loss_type']=='compass_opd':roles+=['reference_scoring','student_anchor_scoring']
    if (receipt.get('status')!='passed' or receipt.get('plan_binding')!=plan['binding']
            or receipt.get('roles')!=roles
            or receipt.get('loss_type')!=plan['training_loss']['loss_type']
            or receipt.get('student_logprob_source')!=plan.get('student_logprob_source')
            or receipt.get('resource_mode')!='shared_two_gpu'
            or receipt.get('phase_handover_test')!='passed'):
        raise ValueError('Training GPU receipt must match this exact plan, roles and loss')
    return receipt


def admitted_pool(plan,visible,token):
    """Resolve UUID visibility against the outer launcher's physical GPU receipt."""
    if not token:raise RuntimeError('Shared two-card admission is missing')
    record=read_json(Path(token));gpus=record.get('gpus',[])
    expected=plan['settings']['gpu_pool']
    if [r['index'] for r in gpus]!=list(expected):
        raise RuntimeError('The outer shared-pool admission selected different physical GPUs')
    values=visible.split(',')
    if values not in ([str(i) for i in expected],[r['uuid'] for r in gpus]):
        raise RuntimeError('CUDA visibility does not match the admitted GPU UUIDs')
    return {'token':token,'gpu_ids':list(expected),'gpu_uuids':[r['uuid'] for r in gpus]}


def validate_acceptance_plan(plan):
    validate_plan(plan)
    s=plan['settings']
    if (plan.get('run_purpose')!='two_step_gpu_acceptance' or s['total_steps']!=2
            or s['train_batch_size']!=2 or plan['training_loss']['loss_type']!='traditional_opd'):
        raise ValueError('Bounded acceptance requires two steps, two questions and traditional OPD')


def validate_formal_plan(plan):
    """Explicit --run authorizes this formal recipe; GPU success is recorded at runtime."""
    validate_plan(plan);s=plan['settings']
    if (plan.get('run_purpose')!='formal_training' or not 1<=s['total_steps']<=64
            or s['train_batch_size']!=32 or s['rollout_n']!=4 or not s['monitor_evaluation']
            or plan.get('data_policy')!='all_train' or plan.get('train_questions')!=1024
            or plan['training_loss']['loss_type']!='traditional_opd'):
        raise ValueError('Fresh formal start requires the authorized 1024-question pool, at most 64 steps and batch32 x4 OPD recipe')


def commit_acceptance(plan):
    """Only produce a passed receipt after the actual trainer and checkpoint complete."""
    root=ROOT/plan['output'];events=[json.loads(s) for s in (root/'logs/training_events.jsonl').read_text().splitlines()]
    updates=[e['step'] for e in events if e['event']=='optimizer_update_committed']
    phases=[e for e in events if e['event']=='phase_end' and e['status']=='completed']
    def count(name):return sum(e['phase']==name for e in phases)
    tracker=root/'checkpoints/latest_checkpointed_iteration.txt'
    if (updates!=[1,2] or count('teacher_scoring_phase')!=2
            or count('student_rollout_logprob_reuse')!=2 or count('student_logprob')!=0
            or plan.get('student_logprob_source')!=STUDENT_LOGPROB_SOURCE
            or count('student_rollout_weight_sync')<2 or count('teacher_sleep_ack')!=2
            or not tracker.is_file() or tracker.read_text().strip()!='2'
            or len(list((root/'checkpoints/global_step_2/actor').glob('optim_world_size_2_rank_*.pt')))!=2):
        raise RuntimeError('Full two-step handover/checkpoint evidence is incomplete')
    receipt={'status':'passed','plan_binding':plan['binding'],'roles':['student_fsdp_rollout','teacher_scoring'],
             'loss_type':'traditional_opd','resource_mode':'shared_two_gpu','phase_handover_test':'passed',
             'actual_optimizer_updates':2,'checkpoint_step':2,'run_purpose':plan['run_purpose']}
    receipt['student_logprob_source']=STUDENT_LOGPROB_SOURCE
    atomic_json(root/'metadata/acceptance_receipt.json',receipt)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--check',action='store_true');mode.add_argument('--prepare',action='store_true')
    mode.add_argument('--run',action='store_true')
    mode.add_argument('--acceptance-run',action='store_true',help='Run exactly two small real OPD batches and save a checkpoint')
    parser.add_argument('--bootstrap-source',action='store_true')
    parser.add_argument('--config',default='configs/ablation.json')
    parser.add_argument('--loss',choices=['traditional_opd','compass_opd'],default='traditional_opd')
    parser.add_argument('--gpu-ids',default='0,1',help='Exactly two physical GPUs shared by all roles')
    parser.add_argument('--wait-seconds',type=int,default=3600,help='Maximum wait for the specified shared GPU pool; 0 fails immediately if busy')
    parser.add_argument('--run-name');parser.add_argument('--teacher-layer',type=int,default=29)
    parser.add_argument('--train-output-root',help='Training output root; default project outputs/, other stages stay in outputs/')
    parser.add_argument('--operator',choices=['slerp','additive'],default='slerp')
    parser.add_argument('--alpha',type=float,default=.1);parser.add_argument('--sv-bank')
    from steer_opd.generation_policy import DEFAULT_MAX_GENERATION_LENGTH,DEFAULT_PRESENCE_PENALTY
    parser.add_argument('--presence-penalty',type=float,default=DEFAULT_PRESENCE_PENALTY,help='Generation penalty for training rollouts and validation; raw OPD/Top16 probabilities remain unwarped')
    parser.add_argument('--anchor-coef',type=float,default=.5)
    parser.add_argument('--reference-model',default='assets/models/Qwen3.5-0.8B')
    parser.add_argument('--batch-size',type=int);parser.add_argument('--total-steps',type=int)
    parser.add_argument('--evaluation-freq',type=int,default=16,help='Evaluate the bare student every N updates and at the final step')
    parser.add_argument('--save-freq',type=int,default=16,help='Save a model checkpoint every N updates and at the final step')
    from steer_opd.retention import ALL_MODELS_LATEST_STATE,CHECKPOINT_POLICIES
    parser.add_argument('--checkpoint-retention',choices=CHECKPOINT_POLICIES,default=ALL_MODELS_LATEST_STATE,
        help='Keep each saved model and only the newest full resume state by default')
    parser.add_argument('--acceptance-receipt')
    parser.add_argument('--overlap-metrics',action='store_true')
    parser.add_argument('--resume-from')
    parser.add_argument('--execute-prepared',help=argparse.SUPPRESS)
    parser.add_argument('--formal-start',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.execute_prepared:
        plan=read_json(args.execute_prepared)
        if args.acceptance_run:validate_acceptance_plan(plan)
        elif args.formal_start:validate_formal_plan(plan)
        else:validate_receipt(plan,args.acceptance_receipt)
        data_manifest=read_json(ROOT/plan['output']/'metadata/data_manifest.json')
        for name,expected in data_manifest['files'].items():
            if hashlib.sha256((ROOT/plan['output']/'data'/name).read_bytes()).hexdigest()!=expected:
                raise ValueError('Prepared training parquet changed')
        if data_manifest['source_sha256']!=plan['input_sha256']:raise ValueError('Training data source binding changed')
        from steer_opd.training.resume_handover import validate_resume_source
        validate_resume_source(plan)
        token=os.environ.get('STEER_OPD_GPU_ADMISSION')
        admission=admitted_pool(plan,os.environ.get('CUDA_VISIBLE_DEVICES',''),token)
        plan={**plan,'explicit_training_run':True,'shared_admission':admission}
        from steer_opd.training.scoring import validate_assets
        if validate_assets(plan)!=plan['assets']:raise ValueError('Weights or steering bank changed')
        config=compose_verl(plan)
        from steer_opd.training.verl_trainer import task_runner_class
        from verl.trainer.main_ppo import run_ppo
        import ray
        from steer_opd.training.telemetry import ResourceMonitor,phase
        logs=ROOT/plan['output']/'logs'
        with ResourceMonitor(logs,admission['gpu_ids'],tensorboard_dir=logs/'tensorboard/system'):
            try:
                with phase(logs,'verl_training_job'):run_ppo(config,task_runner_class=task_runner_class())
                if args.acceptance_run:commit_acceptance(plan)
            finally:
                with phase(logs,'ray_model_process_shutdown'):ray.shutdown()
        return
    if args.bootstrap_source:bootstrap_source()
    settings=VerlSettings(gpu_pool=gpu_ids(args.gpu_ids),
        train_batch_size=args.batch_size if args.batch_size is not None else (2 if args.acceptance_run else 32),
        total_steps=args.total_steps if args.total_steps is not None else (2 if args.acceptance_run else 64),teacher_layer=args.teacher_layer,
        operator=args.operator,alpha=args.alpha,
        presence_penalty=args.presence_penalty,gpu_wait_seconds=args.wait_seconds,
        monitor_evaluation=not args.acceptance_run,monitor_overlap=args.overlap_metrics,evaluation_freq=args.evaluation_freq,
        save_freq=args.save_freq,checkpoint_retention=args.checkpoint_retention,
        run_name=args.run_name or f'verl_{args.loss}_st_T{args.teacher_layer}_{args.operator}{args.alpha:g}_pp{args.presence_penalty:g}_len{DEFAULT_MAX_GENERATION_LENGTH}_seed42_shared2_{"acceptance_minimal_v1" if args.acceptance_run else "formal_minimal_v1"}')
    if args.sv_bank:settings=replace(settings,sv_bank=args.sv_bank)
    loss=TrainingLossConfig(loss_type=args.loss,anchor_coef=args.anchor_coef,reference_model=args.reference_model)
    experiment=read_json(ROOT/args.config)
    if args.train_output_root:experiment['train_output_root']=args.train_output_root
    plan=build_verl_plan(experiment,settings,loss)
    if args.acceptance_run:plan['run_purpose']='two_step_gpu_acceptance'
    else:plan['run_purpose']='formal_training'
    from steer_opd.training.scoring import validate_assets
    plan['assets']=validate_assets(plan)
    if args.resume_from:
        from steer_opd.training.resume_handover import bind_resume_source
        bind_resume_source(plan,ROOT/args.resume_from)
    plan['binding']=digest({k:v for k,v in plan.items() if k!='binding'})
    config=compose_verl(plan)
    print(json.dumps({'plan':plan,'verl_overrides':verl_overrides(plan)},ensure_ascii=False,indent=2),flush=True)
    if not (args.prepare or args.run or args.acceptance_run):return
    if args.run:
        if args.acceptance_receipt:validate_receipt(plan,args.acceptance_receipt)
        else:validate_formal_plan(plan)
    if args.acceptance_run:validate_acceptance_plan(plan)
    if args.acceptance_run or args.run:
        from steer_opd.training.storage import require_training_space
        resuming=bool(plan.get('resume_source')) or (ROOT/plan['output']/'checkpoints/latest_checkpointed_iteration.txt').is_file()
        require_training_space(ROOT/plan['output'],(48 if args.acceptance_run or resuming else 96)*1024**3)
    planfile=freeze_prepare(plan)
    if args.run or args.acceptance_run:
        from steer_opd.experiment_runtime import child_environment
        import uuid
        admission_dir=ROOT/plan['output']/'shared_gpu_admission'/('launch_'+uuid.uuid4().hex[:12])
        command=[sys.executable,'-m','steer_opd.training.idle_launch',
            '--gpu-ids',','.join(str(i) for i in settings.gpu_pool),'--wait-seconds',str(settings.gpu_wait_seconds),
            '--run-dir',str(admission_dir),'--',sys.executable,'-m',
            'steer_opd.training.verl_entry','--execute-prepared',str(planfile)]
        command+=(['--acceptance-run'] if args.acceptance_run else
                  ['--acceptance-receipt',str(Path(args.acceptance_receipt).resolve())] if args.acceptance_receipt else ['--formal-start'])
        subprocess.run(command,cwd=ROOT,env=child_environment(sys.executable),check=True)


if __name__=='__main__':main()
