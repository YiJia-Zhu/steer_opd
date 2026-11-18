"""Process-safe phase timing and bounded resource sampling for training."""
from contextlib import contextmanager
from datetime import datetime
from zoneinfo import ZoneInfo
import fcntl
import json
import os
from pathlib import Path
import subprocess
import threading
import time

import psutil


def stamp():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def resources():
    """CPU seconds and RSS of this process and its currently live descendants."""
    owner=psutil.Process();rows=[]
    for proc in [owner,*owner.children(recursive=True)]:
        try:
            cpu=proc.cpu_times()
            rows.append({'pid':proc.pid,'birth':proc.create_time(),'cpu_seconds':cpu.user+cpu.system,
                         'rss_bytes':proc.memory_info().rss})
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    return {'processes':rows,'cpu_seconds':sum(r['cpu_seconds'] for r in rows),
            'rss_bytes':sum(r['rss_bytes'] for r in rows)}


def event(folder,name,**values):
    if folder is None:return
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    row={'time':stamp(),'event':name,'pid':os.getpid(),**values}
    with (folder/'training_events.jsonl').open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        stream.write(json.dumps(row,ensure_ascii=False,default=str)+'\n');stream.flush()


@contextmanager
def phase(folder,name,**values):
    if folder is None:
        yield
        return
    started=time.monotonic();before=resources()
    event(folder,'phase_start',phase=name,**values)
    print(f'[{stamp()}] {name}: start',flush=True)
    status='failed'
    try:
        yield
        status='completed'
    finally:
        after=resources();seconds=time.monotonic()-started
        event(folder,'phase_end',phase=name,status=status,seconds=seconds,
              cpu_seconds_before=before['cpu_seconds'],cpu_seconds_after=after['cpu_seconds'],
              process_tree_rss_bytes=after['rss_bytes'],**values)
        print(f'[{stamp()}] {name}: {status}, {seconds:.3f}s',flush=True)


class ResourceMonitor:
    """One sampler per admitted job; heartbeat is written to files only."""
    def __init__(self,folder,gpu_ids,interval=15,tensorboard_dir=None):
        self.folder=Path(folder);self.gpu_ids=set(gpu_ids);self.interval=interval
        self.stop=threading.Event();self.thread=None
        self.tensorboard_dir=tensorboard_dir;self.writer=None;self.offset=0;self.step=0
        self.started=time.monotonic()

    def poll_phases(self):
        path=self.folder/'training_events.jsonl'
        if self.writer is None or not path.exists():return
        with path.open() as source:
            source.seek(self.offset)
            while True:
                line=source.readline()
                if not line or not line.endswith('\n'):break
                self.offset=source.tell();row=json.loads(line)
                if 'step' in row:self.step=int(row['step'])
                if row['event']=='phase_end':
                    self.writer.add_scalar('phase_seconds/'+row['phase'],row['seconds'],self.step)

    def sample(self,previous,elapsed):
        now=resources();old={(p['pid'],p['birth']):p['cpu_seconds'] for p in previous.get('processes',[])}
        cpu=sum(max(0,p['cpu_seconds']-old.get((p['pid'],p['birth']),p['cpu_seconds'])) for p in now['processes'])
        gpus=[]
        try:
            import csv
            raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu',
                '--format=csv,noheader,nounits'],text=True,timeout=10)
            gpus=[{'index':int(i),'uuid':u.strip(),'memory_mib':int(m),'utilization_percent':int(v)}
                  for i,u,m,v in csv.reader(raw.splitlines()) if int(i) in self.gpu_ids]
        except (OSError,subprocess.SubprocessError,ValueError):pass
        event(self.folder,'resources',cpu_percent_one_core_100=100*cpu/max(elapsed,.001),
              cpu_scope='driver_and_current_descendants; newly_observed_processes_start_next_interval',
              rss_bytes=now['rss_bytes'],processes=now['processes'],gpus=gpus)
        if self.writer is not None:
            tick=int(time.monotonic()-self.started)
            self.writer.add_scalar('system/cpu_percent_one_core_100',100*cpu/max(elapsed,.001),tick)
            self.writer.add_scalar('system/process_tree_rss_gib',now['rss_bytes']/2**30,tick)
            for gpu in gpus:
                self.writer.add_scalar(f'system/gpu{gpu["index"]}_memory_gib',gpu['memory_mib']/1024,tick)
                self.writer.add_scalar(f'system/gpu{gpu["index"]}_utilization',gpu['utilization_percent'],tick)
            self.poll_phases();self.writer.flush()
        return now

    def __enter__(self):
        if self.tensorboard_dir is not None:
            from torch.utils.tensorboard import SummaryWriter
            self.writer=SummaryWriter(str(self.tensorboard_dir),flush_secs=10)
        def loop():
            previous=resources();last=time.monotonic()
            while not self.stop.wait(self.interval):
                tick=time.monotonic();previous=self.sample(previous,tick-last);last=tick
        self.thread=threading.Thread(target=loop,daemon=True);self.thread.start()
        event(self.folder,'job_started',gpu_ids=sorted(self.gpu_ids))
        return self

    def __exit__(self,kind,error,tb):
        self.stop.set();self.thread.join(timeout=12)
        event(self.folder,'job_finished',status='failed' if error else 'completed',error=repr(error) if error else None)
        if self.writer is not None:
            self.poll_phases();self.writer.close()
