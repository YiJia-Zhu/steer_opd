"""Synchronous batch-scoring client; inference runs in the isolated EasySteer env."""
import os
import fcntl
from pathlib import Path
import subprocess
import time

import torch

from steer_opd.direction_bank import file_sha256
from steer_opd.experiment_runtime import child_environment
from steer_opd.resumable_core import ROOT, atomic_json, read_json
from steer_opd.training.scoring import codecs_for_plan, prepare_request


class ScorerClient:
    def __init__(self,plan):
        self.plan=plan;self.folder=ROOT/plan['output']/'scorer'
        self.codecs=codecs_for_plan(plan);self.process=None;self.log=None;self.lock=None

    def start(self):
        self.folder.mkdir(parents=True,exist_ok=True)
        self.lock=(self.folder/'client.lock').open('a')
        fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        service=(self.folder/'service.lock').open('a')
        try:fcntl.flock(service,fcntl.LOCK_EX|fcntl.LOCK_NB)
        finally:service.close()
        from steer_opd.training.scorer_service import process_ticks
        if (self.folder/'ready.json').exists():
            owner=read_json(self.folder/'owner.json')
            if process_ticks(owner['pid'])==owner['start_ticks']:
                raise RuntimeError('This training run already owns a live scorer; refuse duplicate launch')
            (self.folder/'ready.json').unlink()
        atomic_json(self.folder/'owner.json',{'pid':os.getpid(),'start_ticks':process_ticks(os.getpid())})
        planfile=self.folder/'plan.json'
        if planfile.exists() and read_json(planfile)!=self.plan:raise ValueError('Frozen scorer plan changed')
        atomic_json(planfile,self.plan)
        python=str(ROOT/'.venv-easysteer/bin/python')
        env=child_environment(python)
        pool=','.join(str(v) for v in self.plan['settings']['teacher_gpus'])
        command=[str(ROOT/'.venv-opd/bin/python'),str(ROOT/'scripts/run_on_idle_gpus.py'),
            '--count','2','--gpu-ids',pool,'--wait-seconds',str(self.plan['settings']['gpu_wait_seconds']),
            '--run-dir',str(self.folder/'admission'/str(time.time_ns())),'--',python,
            '-m','steer_opd.training.scorer_service','--plan',str(planfile),'--folder',str(self.folder)]
        self.log=(self.folder/'supervisor.log').open('a')
        self.process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=self.log,stderr=subprocess.STDOUT)
        self._wait(lambda:(self.folder/'ready.json').exists(),timeout=self.plan['settings']['gpu_wait_seconds']+2800)
        if read_json(self.folder/'ready.json')['plan_binding']!=self.plan['binding']:raise ValueError('Wrong scorer service identity')

    def _wait(self,predicate,*,timeout=None):
        started=time.monotonic()
        while not predicate():
            if self.process.poll() is not None:raise RuntimeError(f'EasySteer scorer exited; see {self.folder}/supervisor.log')
            if time.monotonic()-started>(timeout or self.plan['settings']['scorer_timeout']):raise TimeoutError('Frozen scoring batch timed out')
            time.sleep(.2)

    def score(self,rows):
        request=prepare_request(self.plan,rows,self.codecs);key=request['request_id']
        atomic_json(self.folder/'requests'/f'{key}.json',request)
        roles=['teacher']+(['reference','student_anchor'] if self.plan['training_loss']['loss_type']=='compass_opd' else [])
        output=[{'alignment':r['alignment']} for r in request['rows']]
        for role in roles:
            folder=self.folder/'cache'/role/key
            self._wait(lambda:(folder/'complete.json').exists())
            receipt=read_json(folder/'complete.json')
            if (receipt['request_id']!=key or receipt['plan_binding']!=self.plan['binding']
                    or receipt['source_ids']!=[r['id'] for r in rows]
                    or receipt['sha256']!=file_sha256(folder/'observed_logps.pt')
                    or receipt['assets']!=self.plan['assets']):
                raise ValueError('Frozen scorer reply identity changed')
            values=torch.load(folder/'observed_logps.pt',map_location='cpu',weights_only=True)
            if len(values)!=len(output):raise ValueError('Scoring reply row count changed')
            name='student_anchor_logps' if role=='student_anchor' else role+'_logps'
            for row,value in zip(output,values):row[name]=value
        return output

    def close(self):
        if self.process is not None:
            atomic_json(self.folder/'stop.json',{'reason':'training_driver_finished'})
            try:self.process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:self.process.wait(timeout=45)
                except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
        if self.log:self.log.close()
        if self.lock:self.lock.close()
