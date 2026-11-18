"""Sleep-enabled frozen scorers sharing the one admitted two-GPU training pool."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import torch

from steer_opd.resumable_core import ROOT,atomic_json,atomic_torch,digest,read_json
from steer_opd.direction_bank import file_sha256
from steer_opd.training.telemetry import phase
from steer_opd.training.scoring_request import SCHEMA as REQUEST_SCHEMA,read_native_request
from steer_opd.training.retention import batch_root
from steer_opd.training.paths import training_path


def process_ticks(pid):
    try:return int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19])
    except (FileNotFoundError,ProcessLookupError):return None


def check_owner(folder):
    owner=read_json(folder/'owner.json')
    if (folder/'stop.json').exists() or process_ticks(owner['pid'])!=owner['start_ticks']:
        raise KeyboardInterrupt('Training owner exited')


def run_role(plan,folder,role):
    from transformers import AutoTokenizer
    from steer_opd.compass_alignment import ByteLevelCodec
    from steer_opd.easysteer_static import fixed_prefix_scores,build_spec
    from steer_opd.static_screen import candidate_vectors
    from steer_opd.training.scoring import validate_assets
    from steer_opd.training.sleep_engine import create_sleep_engine,FrozenEngineResidency
    assets=validate_assets(plan)
    if assets!=plan['assets']:raise ValueError('Frozen scoring weights, tokenizer or SV changed')
    s=plan['settings']
    path=(plan['teacher'] if role=='teacher' else plan['student'] if role=='student_anchor'
          else plan['training_loss']['reference_model'])
    tokenizer_path=plan['tokenizer_paths']['teacher'] if role=='teacher' else path
    codec=ByteLevelCodec(AutoTokenizer.from_pretrained(ROOT/tokenizer_path,local_files_only=True))
    memory={'teacher':s['teacher_memory'],'reference':s['reference_memory'],'student_anchor':s['anchor_memory']}[role]
    logs=ROOT/plan['output']/'logs'
    with phase(logs,f'{role}_model_initialization'):
        engine=create_sleep_engine(ROOT/path,codec.support_size,max_model_len=s['score_max_model_len'],
            tensor_parallel_size=2 if role=='teacher' else 1,memory=memory,batch_invariant=role=='student_anchor')
    residency=FrozenEngineResidency(engine)
    vectors=None
    if role=='teacher':
        bank=torch.load(ROOT/s['sv_bank'],map_location='cpu',weights_only=True)
        condition={'direction':'S-T','granularity':'global','operator':s['operator'],
                   'alpha':s['alpha'],'layers':[s['teacher_layer']]}
        if s['operator']=='additive':condition['additive_scaling']='raw_mean_difference_v1'
        vectors,_=candidate_vectors(bank,{'id':'global'},condition)
    with phase(logs,f'{role}_initial_sleep'):residency.sleep()
    atomic_json(folder/('ready_'+role+'.json'),{'status':'ready','state':'asleep','pid':os.getpid(),'plan_binding':plan['binding']})
    previous=None;pending=None
    while True:
        check_owner(folder)
        control=folder/'control'/f'{role}.json'
        if control.exists():
            command=read_json(control)
            if command['plan_binding']!=plan['binding']:raise ValueError('Wrong scoring control identity')
            if command['command_id']!=previous:
                if command['action']=='wake':
                    pending=command['request_id']
                    with phase(logs,f'{role}_model_wake'):residency.wake()
                elif command['action']=='sleep':
                    with phase(logs,f'{role}_model_sleep'):residency.sleep()
                    pending=None
                else:raise ValueError('Unknown shared-pool control action')
                previous=command['command_id']
                atomic_json(folder/'acks'/f'{previous}.json',{'command_id':previous,'state':residency.state,
                    'role':role,'plan_binding':plan['binding']})
        if residency.state=='awake' and pending:
            with phase(logs,'cpu_scoring_request_read',role=role):
                request,rows=read_native_request(batch_root(folder,plan)/'requests',pending,plan['binding'],role)
            target=batch_root(folder,plan)/'cache'/role/pending;done=target/'complete.json'
            if not done.exists():
                if any(len(r['prompt_token_ids'])+len(r['token_ids'])+1>s['score_max_model_len'] for r in rows):
                    raise ValueError('Native scoring context exceeded; never truncate a student trajectory silently')
                usable=[i for i,r in enumerate(rows) if r['token_ids']];active=[rows[i] for i in usable]
                specs=[build_spec(r,[True]*len(r['token_ids']),vectors,operator=s['operator'],alpha=s['alpha'])
                       if role=='teacher' else False for r in active]
                with phase(logs,f'{role}_model_scoring',trajectories=len(active)):
                    scores=fixed_prefix_scores(engine,active,specs,include_sampled=True) if active else []
                values=[torch.empty(0) for _ in rows]
                for i,score in zip(usable,scores):values[i]=score['sampled_logps'].float().cpu()
                if role=='teacher' and s.get('monitor_overlap'):
                    from steer_opd.training.overlap_monitor import save_top16
                    top16=[{'ids':torch.empty((0,16),dtype=torch.int32),'logps':torch.empty((0,16)),
                            'sampled_logps':torch.empty(0)} for _ in rows]
                    for i,score in zip(usable,scores):top16[i]=score
                    save_top16(target/'top16.pt.gz',top16)
                    atomic_json(target/'top16.complete.json',{'request_id':pending,'request_schema':REQUEST_SCHEMA,
                        'plan_binding':plan['binding'],
                        'sha256':file_sha256(target/'top16.pt.gz'),'assets':assets,
                        'source_ids':request['source_ids']})
                atomic_torch(target/'observed_logps.pt',values)
                atomic_json(done,{'request_id':pending,'request_schema':REQUEST_SCHEMA,
                    'plan_binding':plan['binding'],'role':role,
                    'sha256':file_sha256(target/'observed_logps.pt'),
                    'source_ids':request['source_ids'],'assets':assets})
            pending=None
        time.sleep(.05)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True);parser.add_argument('--folder',required=True)
    parser.add_argument('--role',choices=['teacher','reference','student_anchor'])
    args=parser.parse_args();plan=read_json(args.plan);folder=training_path(args.folder)
    if plan.get('scoring_request_schema')!=REQUEST_SCHEMA:
        raise ValueError('Prepare a new training plan for array-backed scoring requests')
    if folder!=(ROOT/plan['output']/'scorer').resolve():raise ValueError('Scorer directory differs from training plan')
    if not os.environ.get('STEER_OPD_GPU_ADMISSION'):raise RuntimeError('One shared-pool GPU admission is required')
    if not plan.get('explicit_training_run') or plan['resource_mode']!='shared_two_gpu':
        raise RuntimeError('Expected an explicitly accepted shared-two-card training plan')
    if args.role:return run_role(plan,folder,args.role)
    folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'service.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (folder/'stop.json').unlink(missing_ok=True)
    roles=['teacher']+(['reference','student_anchor'] if plan['training_loss']['loss_type']=='compass_opd' else [])
    children=[];births=[];logs=[]
    def interrupted(signum,frame):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,interrupted)
    visible=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    if len(visible)!=2:raise ValueError('Exactly two GPUs must have been admitted for the complete job')
    try:
        # Initialize one role at a time; each reports asleep before the next
        # role initializes. Only one frozen role may subsequently be awake.
        for role in roles:
            (folder/('ready_'+role+'.json')).unlink(missing_ok=True)
            # Resume must never replay a previous process's last wake command
            # during initialization, before the student has handed over GPUs.
            (folder/'control'/f'{role}.json').unlink(missing_ok=True)
            log=(folder/(role+'.log')).open('a');logs.append(log)
            env={**os.environ,'CUDA_VISIBLE_DEVICES':','.join(visible if role=='teacher' else visible[:1])}
            child=subprocess.Popen([sys.executable,'-m','steer_opd.training.scorer_service','--plan',args.plan,
                '--folder',str(folder),'--role',role],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            children.append(child);births.append(process_ticks(child.pid))
            atomic_json(folder/'role_processes.json',[{'pid':c.pid,'start_ticks':b} for c,b in zip(children,births)])
            started=time.monotonic()
            while not (folder/('ready_'+role+'.json')).exists():
                check_owner(folder)
                if child.poll() is not None:raise RuntimeError(f'{role} scorer exited; see {folder/(role+".log")}')
                if time.monotonic()-started>900:raise TimeoutError(f'{role} scorer initialization timed out')
                time.sleep(.2)
        atomic_json(folder/'ready.json',{'roles':roles,'state':'asleep','plan_binding':plan['binding'],'pid':os.getpid()})
        while True:
            check_owner(folder)
            if any(p.poll() is not None for p in children):raise RuntimeError('A frozen scorer exited unexpectedly')
            time.sleep(.2)
    except KeyboardInterrupt:pass
    finally:
        from steer_opd.training.process_cleanup import stop_session
        for child,birth in zip(children,births):stop_session(child.pid,birth,child=child)
        for log in logs:log.close()
        (folder/'ready.json').unlink(missing_ok=True)


if __name__=='__main__':main()
