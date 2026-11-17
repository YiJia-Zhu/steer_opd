"""Persistent project-local EasySteer scorers, launched only after GPU admission."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import torch

from steer_opd.resumable_core import ROOT, atomic_json, atomic_torch, digest, read_json
from steer_opd.direction_bank import file_sha256


def run_role(plan, folder, role):
    from transformers import AutoTokenizer
    from steer_opd.compass_alignment import ByteLevelCodec
    from steer_opd.easysteer_static import create_engine, fixed_prefix_scores, build_spec
    from steer_opd.static_screen import candidate_vectors
    from steer_opd.training.scoring import validate_assets
    assets=validate_assets(plan)
    if assets!=plan['assets']:raise ValueError('Frozen scoring weights or SV changed')
    s=plan['settings']
    path=(plan['teacher'] if role=='teacher' else plan['student'] if role=='student_anchor'
          else plan['training_loss']['reference_model'])
    tokenizer_path=plan['tokenizer_paths']['teacher'] if role=='teacher' else path
    codec=ByteLevelCodec(AutoTokenizer.from_pretrained(ROOT/tokenizer_path,local_files_only=True))
    memory={'teacher':s['teacher_memory'],'reference':s['reference_memory'],'student_anchor':s['anchor_memory']}[role]
    engine=create_engine(ROOT/path,codec.support_size,max_model_len=s['score_max_model_len'],
        tensor_parallel_size=2 if role=='teacher' else 1,gpu_memory_utilization=memory,
        batch_invariant=role=='student_anchor',seed=42)
    vectors=None
    if role=='teacher':
        bank=torch.load(ROOT/s['sv_bank'],map_location='cpu',weights_only=True)
        condition={'direction':'S-T','granularity':'global','operator':s['operator'],
                   'alpha':s['alpha'],'layers':[s['teacher_layer']]}
        if s['operator']=='additive':condition['additive_scaling']='raw_mean_difference_v1'
        vectors,_=candidate_vectors(bank,{'id':'global'},condition)
    atomic_json(folder/('ready_'+role+'.json'),{'status':'ready','pid':os.getpid(),'plan_binding':plan['binding']})
    while not (folder/'stop.json').exists():
        for file in sorted((folder/'requests').glob('*.json')):
            request=read_json(file)
            if request['plan_binding']!=plan['binding']:raise ValueError('Scorer request belongs to another training run')
            check={k:v for k,v in request.items() if k!='request_id'}
            if request['request_id']!=digest(check):raise ValueError('Scorer request identity changed')
            target=folder/'cache'/role/request['request_id'];done=target/'complete.json'
            if done.exists():continue
            rows=[r['native'][role] for r in request['rows']]
            if any(len(r['prompt_token_ids'])+len(r['token_ids'])+1>s['score_max_model_len'] for r in rows):
                raise ValueError('Native scoring context exceeded; response is never silently truncated')
            usable=[i for i,r in enumerate(rows) if r['token_ids']]
            active=[rows[i] for i in usable]
            specs=[build_spec(r,[True]*len(r['token_ids']),vectors,operator=s['operator'],alpha=s['alpha'])
                   if role=='teacher' else False for r in active]
            scores=fixed_prefix_scores(engine,active,specs,include_sampled=True) if active else []
            values=[torch.empty(0) for _ in rows]
            for i,score in zip(usable,scores):values[i]=score['sampled_logps'].float().cpu()
            atomic_torch(target/'observed_logps.pt',values)
            atomic_json(done,{'request_id':request['request_id'],'plan_binding':plan['binding'],
                'role':role,'sha256':file_sha256(target/'observed_logps.pt'),
                'source_ids':[r['source']['id'] for r in request['rows']], 'assets':assets})
        time.sleep(.1)


def process_ticks(pid):
    try:return int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19])
    except (FileNotFoundError,ProcessLookupError):return None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True);parser.add_argument('--folder',required=True)
    parser.add_argument('--role',choices=['teacher','reference','student_anchor'])
    args=parser.parse_args();plan=read_json(args.plan);folder=Path(args.folder).resolve()
    if not folder.is_relative_to(ROOT):raise ValueError('Scorer files must stay in project')
    if not os.environ.get('STEER_OPD_GPU_ADMISSION'):raise RuntimeError('Idle GPU admission is required')
    if not plan.get('explicit_training_run'):raise RuntimeError('Scorer requires an explicit accepted training run')
    if args.role:return run_role(plan,folder,args.role)
    folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'service.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (folder/'stop.json').unlink(missing_ok=True)
    roles=['teacher']+(['reference','student_anchor'] if plan['training_loss']['loss_type']=='compass_opd' else [])
    children=[];logs=[]
    def interrupted(signum,frame):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,interrupted)
    visible=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    try:
        # Each role uses a distinct process. Smaller scorers share the admitted
        # teacher pool, with explicit fractions; no CUDA device visibility mutation.
        for role in roles:
            (folder/('ready_'+role+'.json')).unlink(missing_ok=True)
            log=(folder/(role+'.log')).open('a');logs.append(log)
            env={**os.environ,'CUDA_VISIBLE_DEVICES':','.join(visible if role=='teacher' else visible[:1])}
            child=subprocess.Popen([sys.executable,'-m','steer_opd.training.scorer_service','--plan',args.plan,
                '--folder',str(folder),'--role',role],env=env,stdout=log,stderr=subprocess.STDOUT)
            children.append(child)
            started=time.monotonic()
            while not (folder/('ready_'+role+'.json')).exists():
                owner=read_json(folder/'owner.json')
                if (folder/'stop.json').exists() or process_ticks(owner['pid'])!=owner['start_ticks']:
                    raise KeyboardInterrupt('Training owner exited during scorer initialization')
                if child.poll() is not None:raise RuntimeError(f'{role} scorer exited; see {folder/(role+".log")}')
                if time.monotonic()-started>900:raise TimeoutError(f'{role} scorer initialization timed out')
                time.sleep(.2)
        atomic_json(folder/'ready.json',{'roles':roles,'plan_binding':plan['binding'],'pid':os.getpid()})
        while not (folder/'stop.json').exists():
            owner=read_json(folder/'owner.json')
            if process_ticks(owner['pid'])!=owner['start_ticks']:break
            if any(p.poll() is not None for p in children):raise RuntimeError('A frozen scorer exited unexpectedly')
            time.sleep(.2)
    finally:
        for p in children:
            if p.poll() is None:p.terminate()
        for p in children:
            try:p.wait(timeout=30)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        for log in logs:log.close()
        (folder/'ready.json').unlink(missing_ok=True)


if __name__=='__main__':main()
