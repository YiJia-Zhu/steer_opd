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
    partition=read_json(source/'split.json')
    for split in ('select','confirm'):
        records=[]
        for qid in partition[split+'_ids']:
            row=questions[str(qid)]
            records.append({'data_source':'steer_opd','prompt':[{'role':'user','content':row['question']+TASK_SUFFIX}],
                'ability':'math','reward_model':{'style':'rule','ground_truth':''},
                'extra_info':{'question_id':str(qid),'question':row['question']}})
        output=root/'data'/('train.parquet' if split=='select' else 'confirm.parquet')
        output.parent.mkdir(parents=True,exist_ok=True)
        if output.exists():
            old=pd.read_parquet(output).to_json(orient='records')
            if json.loads(old)!=json.loads(pd.DataFrame(records).to_json(orient='records')):
                raise ValueError('Frozen training dataset changed')
        else:pd.DataFrame(records).to_parquet(output,index=False)
    files={name:hashlib.sha256((root/'data'/name).read_bytes()).hexdigest()
           for name in ('train.parquet','confirm.parquet')}
    manifest={'source_sha256':plan['input_sha256'],'files':files,'train_questions':700,'confirm_questions':300}
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
            or receipt.get('loss_type')!=plan['training_loss']['loss_type']):
        raise ValueError('Training GPU receipt must match this exact plan, roles and loss')
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--check',action='store_true');mode.add_argument('--prepare',action='store_true')
    mode.add_argument('--run',action='store_true')
    parser.add_argument('--bootstrap-source',action='store_true')
    parser.add_argument('--config',default='configs/ablation.json')
    parser.add_argument('--loss',choices=['traditional_opd','compass_opd'],default='traditional_opd')
    parser.add_argument('--student-gpus',default='0,1');parser.add_argument('--teacher-gpus',default='2,3')
    parser.add_argument('--run-name');parser.add_argument('--teacher-layer',type=int,default=29)
    parser.add_argument('--operator',choices=['slerp','additive'],default='slerp')
    parser.add_argument('--alpha',type=float,default=.1);parser.add_argument('--sv-bank')
    parser.add_argument('--anchor-coef',type=float,default=.5)
    parser.add_argument('--reference-model',default='assets/models/Qwen3.5-0.8B')
    parser.add_argument('--batch-size',type=int,default=16);parser.add_argument('--total-steps',type=int,default=100)
    parser.add_argument('--acceptance-receipt')
    parser.add_argument('--execute-prepared',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.execute_prepared:
        plan=read_json(args.execute_prepared)
        validate_receipt(plan,args.acceptance_receipt)
        data_manifest=read_json(ROOT/plan['output']/'metadata/data_manifest.json')
        for name,expected in data_manifest['files'].items():
            if hashlib.sha256((ROOT/plan['output']/'data'/name).read_bytes()).hexdigest()!=expected:
                raise ValueError('Prepared training parquet changed')
        if data_manifest['source_sha256']!=plan['input_sha256']:raise ValueError('Training data source binding changed')
        plan={**plan,'explicit_training_run':True}
        from steer_opd.training.scoring import validate_assets
        if validate_assets(plan)!=plan['assets']:raise ValueError('Weights or steering bank changed')
        config=compose_verl(plan)
        from steer_opd.training.verl_trainer import task_runner_class
        from verl.trainer.main_ppo import run_ppo
        run_ppo(config,task_runner_class=task_runner_class())
        return
    if args.bootstrap_source:bootstrap_source()
    settings=VerlSettings(student_gpus=gpu_ids(args.student_gpus),teacher_gpus=gpu_ids(args.teacher_gpus),
        train_batch_size=args.batch_size,total_steps=args.total_steps,teacher_layer=args.teacher_layer,
        operator=args.operator,alpha=args.alpha,
        run_name=args.run_name or f'verl_{args.loss}_st_T{args.teacher_layer}_{args.operator}{args.alpha:g}_seed42_v1')
    if args.sv_bank:settings=replace(settings,sv_bank=args.sv_bank)
    loss=TrainingLossConfig(loss_type=args.loss,anchor_coef=args.anchor_coef,reference_model=args.reference_model)
    plan=build_verl_plan(read_json(ROOT/args.config),settings,loss)
    from steer_opd.training.scoring import validate_assets
    plan['assets']=validate_assets(plan)
    plan['binding']=digest({k:v for k,v in plan.items() if k!='binding'})
    config=compose_verl(plan)
    print(json.dumps({'plan':plan,'verl_overrides':verl_overrides(plan)},ensure_ascii=False,indent=2),flush=True)
    if not (args.prepare or args.run):return
    if args.run:validate_receipt(plan,args.acceptance_receipt)
    planfile=freeze_prepare(plan)
    if args.run:
        from steer_opd.experiment_runtime import child_environment
        from omegaconf import OmegaConf
        command=[sys.executable,str(ROOT/'scripts/run_on_idle_gpus.py'),'--count',str(len(settings.student_gpus)),
            '--gpu-ids',','.join(str(i) for i in settings.student_gpus),'--wait-seconds',str(settings.gpu_wait_seconds),
            '--run-dir',str(ROOT/plan['output']/'student_admission'),'--',sys.executable,'-m',
            'steer_opd.training.verl_entry','--execute-prepared',str(planfile),'--acceptance-receipt',str(Path(args.acceptance_receipt).resolve())]
        subprocess.run(command,cwd=ROOT,env=child_environment(sys.executable),check=True)


if __name__=='__main__':main()
