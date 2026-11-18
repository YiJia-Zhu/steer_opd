"""Sleep/wake batch client inside the single already-admitted two-card pool."""
import os
import fcntl
from pathlib import Path
import signal
import subprocess
import time
import uuid

import torch

from steer_opd.direction_bank import file_sha256
from steer_opd.experiment_runtime import child_environment
from steer_opd.resumable_core import ROOT,atomic_json,read_json
from steer_opd.training.scoring import codecs_for_plan,prepare_request
from steer_opd.training.scoring_request import SCHEMA as REQUEST_SCHEMA,write_request
from steer_opd.training.telemetry import phase
from steer_opd.training.retention import batch_root,reset_batch_workspace
from steer_opd.retention import minimal


class ScorerClient:
    def __init__(self,plan):
        self.plan=plan;self.folder=ROOT/plan['output']/'scorer'
        self.codecs=codecs_for_plan(plan);self.process=None;self.log=None;self.lock=None
        self.roles=['teacher']+(['reference','student_anchor'] if plan['training_loss']['loss_type']=='compass_opd' else [])
        self.states={role:'unknown' for role in self.roles};self.shared_phase=False;self.start_ticks=None
        self.pending_request=None

    def start(self):
        if self.plan.get('scoring_request_schema') != REQUEST_SCHEMA:
            raise ValueError('Prepare a new training plan for array-backed scoring requests')
        admission=self.plan.get('shared_admission')
        if (not admission or not admission.get('token')
                or set(admission['gpu_ids'])!=set(self.plan['settings']['gpu_pool'])):
            raise RuntimeError('Frozen scorers must inherit the one whole-job two-card admission')
        self.folder.mkdir(parents=True,exist_ok=True)
        self.lock=(self.folder/'client.lock').open('a');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        service=(self.folder/'service.lock').open('a')
        try:fcntl.flock(service,fcntl.LOCK_EX|fcntl.LOCK_NB)
        finally:service.close()
        from steer_opd.training.scorer_service import process_ticks
        if (self.folder/'ready.json').exists():
            owner=read_json(self.folder/'owner.json')
            if process_ticks(owner['pid'])==owner['start_ticks']:raise RuntimeError('This training run owns a live scorer')
            (self.folder/'ready.json').unlink()
        atomic_json(self.folder/'owner.json',{'pid':os.getpid(),'start_ticks':process_ticks(os.getpid())})
        planfile=self.folder/'plan.json'
        stable_plan={k:v for k,v in self.plan.items() if k!='shared_admission'}
        if planfile.exists() and read_json(planfile)!=stable_plan:raise ValueError('Frozen scorer plan changed')
        atomic_json(planfile,stable_plan)
        reset_batch_workspace(self.folder,self.plan)
        atomic_json(self.folder/'runtime_admission.json',admission)
        python=str(ROOT/'.venv-easysteer/bin/python');env=child_environment(python)
        env.update(CUDA_VISIBLE_DEVICES=','.join(str(v) for v in admission['gpu_ids']),
                   STEER_OPD_GPU_ADMISSION=admission['token'])
        # No second idle-card admission: the outer training launcher owns both
        # physical GPUs for VERL and all scoring subprocesses for the entire run.
        command=[python,'-m','steer_opd.training.scorer_service','--plan',str(planfile),'--folder',str(self.folder)]
        self.log=(self.folder/'supervisor.log').open('a')
        self.process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=self.log,stderr=subprocess.STDOUT,start_new_session=True)
        self.start_ticks=process_ticks(self.process.pid)
        with phase(ROOT/self.plan['output']/'logs','frozen_scorer_service_initialization'):
            self._wait(lambda:(self.folder/'ready.json').exists(),timeout=2800)
        ready=read_json(self.folder/'ready.json')
        if ready['plan_binding']!=self.plan['binding'] or ready['state']!='asleep' or ready['roles']!=self.roles:
            raise ValueError('Frozen scoring engines did not all acknowledge initial sleep')
        self.states={role:'asleep' for role in self.roles}

    def _wait(self,predicate,*,timeout=None):
        started=time.monotonic()
        while True:
            if self.process.poll() is not None:raise RuntimeError(f'Frozen scorer exited; see {self.folder}/supervisor.log')
            if predicate():return
            if time.monotonic()-started>(timeout or self.plan['settings']['scorer_timeout']):raise TimeoutError('Frozen scoring operation timed out')
            time.sleep(.1)

    def ensure_asleep(self):
        if any(v!='asleep' for v in self.states.values()):
            raise RuntimeError('A frozen scorer has not acknowledged sleep; refuse student GPU work')

    def begin_shared_phase(self):
        self.ensure_asleep();self.shared_phase=True

    def end_shared_phase(self):
        self.ensure_asleep();self.shared_phase=False

    def control(self,role,action,request_id=None):
        if action=='wake':
            if not self.shared_phase:raise RuntimeError('No confirmed student offload for frozen wake')
            self.ensure_asleep()
        command_id=uuid.uuid4().hex
        self.states[role]='transitioning'
        atomic_json(self.folder/'control'/f'{role}.json',{'command_id':command_id,'action':action,
            'request_id':request_id,'plan_binding':self.plan['binding']})
        path=self.folder/'acks'/f'{command_id}.json'
        with phase(ROOT/self.plan['output']/'logs',f'{role}_{action}_ack'):
            self._wait(path.exists)
        result=read_json(path);expected='awake' if action=='wake' else 'asleep'
        if result!={'command_id':command_id,'state':expected,'role':role,'plan_binding':self.plan['binding']}:
            raise ValueError('Shared GPU handover acknowledgement changed')
        self.states[role]=expected

    def score(self,rows):
        if not self.shared_phase:raise RuntimeError('Scoring requires a confirmed shared-pool handover')
        self.ensure_asleep()
        with phase(ROOT/self.plan['output']/'logs','cpu_tokenizer_span_alignment',trajectories=len(rows)):
            request=prepare_request(self.plan,rows,self.codecs)
        with phase(ROOT/self.plan['output']/'logs','cpu_scoring_request_commit',trajectories=len(rows)):
            manifest=write_request(batch_root(self.folder,self.plan)/'requests',request)
        key=manifest['request_id']
        if minimal(self.plan) and self.pending_request is not None and self.pending_request['request_id']!=key:
            raise RuntimeError('Previous scoring batch has not completed its update')
        self.pending_request={'request_id':key,'payload_sha256':manifest['payload']['sha256']}
        output=[{'alignment':r['alignment']} for r in request['rows']]
        for role in self.roles:
            folder=batch_root(self.folder,self.plan)/'cache'/role/key
            if not (folder/'complete.json').exists():
                try:
                    self.control(role,'wake',key)
                    self._wait(lambda:(folder/'complete.json').exists())
                finally:
                    # Wake or scoring failure never permits a subsequent student
                    # phase unless the blocking sleep acknowledgement succeeds.
                    self.control(role,'sleep')
            receipt=read_json(folder/'complete.json')
            if (receipt.get('request_schema')!=REQUEST_SCHEMA
                    or receipt['request_id']!=key or receipt['plan_binding']!=self.plan['binding']
                    or receipt['source_ids']!=[r['id'] for r in rows]
                    or receipt['sha256']!=file_sha256(folder/'observed_logps.pt')
                    or receipt['assets']!=self.plan['assets']):raise ValueError('Frozen scorer reply identity changed')
            values=torch.load(folder/'observed_logps.pt',map_location='cpu',weights_only=True)
            if len(values)!=len(output):raise ValueError('Scoring reply row count changed')
            name='student_anchor_logps' if role=='student_anchor' else role+'_logps'
            for row,value in zip(output,values):row[name]=value
            if role=='teacher' and self.plan['settings'].get('monitor_overlap'):
                from steer_opd.training.overlap_monitor import load_top16
                proof=read_json(folder/'top16.complete.json')
                if (proof.get('request_schema')!=REQUEST_SCHEMA
                        or proof['request_id']!=key or proof['plan_binding']!=self.plan['binding']
                        or proof['assets']!=self.plan['assets'] or proof['source_ids']!=[r['id'] for r in rows]
                        or proof['sha256']!=file_sha256(folder/'top16.pt.gz')):
                    raise ValueError('Teacher Top16 does not belong to this same scoring call')
                top16=load_top16(folder/'top16.pt.gz')
                if len(top16)!=len(values):raise ValueError('Teacher Top16 changed row count')
                for row,value,score in zip(output,values,top16):
                    if not torch.equal(value,score['sampled_logps'].float()):
                        raise ValueError('Teacher Top16 and OPD observed logprobs differ')
                    row['teacher_top16']=score
        self.ensure_asleep()
        return output

    def release_batch(self,step):
        """Release only after a finite optimizer update and scalar commit."""
        if not minimal(self.plan):return
        self.ensure_asleep()
        if self.shared_phase:raise RuntimeError('Cannot retire an active scoring batch')
        if self.pending_request is None:raise RuntimeError('No scored batch to retire')
        from steer_opd.training.telemetry import event
        stats=reset_batch_workspace(self.folder,self.plan)
        event(ROOT/self.plan['output']/'logs','batch_artifacts_retired',step=step,
              **self.pending_request,**stats)
        self.pending_request=None

    def close(self):
        from steer_opd.training.process_cleanup import stop_session
        if self.process is not None:
            atomic_json(self.folder/'stop.json',{'reason':'training_driver_finished'})
            stop_session(self.process.pid,self.start_ticks,child=self.process,grace=45)
            ledger=self.folder/'role_processes.json'
            if ledger.exists():
                for role in read_json(ledger):stop_session(role['pid'],role['start_ticks'])
        if self.log:self.log.close()
        if self.lock:self.lock.close()
