"""Admit one two-card training job without stopping independent experiment jobs."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from steer_opd.resumable_core import ROOT,atomic_json,read_json
from steer_opd.experiment_runtime import child_environment
from steer_opd.training.process_cleanup import cleanup_api
from steer_opd.training.telemetry import event,phase
from steer_opd.training.paths import training_path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-ids',required=True);parser.add_argument('--run-dir',required=True)
    parser.add_argument('--wait-seconds',type=int,default=3600)
    parser.add_argument('command',nargs=argparse.REMAINDER);args=parser.parse_args()
    pool=[int(v) for v in args.gpu_ids.split(',')]
    if len(pool)!=2 or len(set(pool))!=2:raise ValueError('Exactly two distinct physical cards required')
    command=args.command[1:] if args.command[:1]==['--'] else args.command
    if not command:raise ValueError('Missing training command')
    folder=training_path(args.run_dir)
    folder.mkdir(parents=True,exist_ok=True);record_path=folder/'process.json'
    if record_path.exists():raise FileExistsError('Use a new immutable run identity')
    api=cleanup_api();deadline=time.monotonic()+args.wait_seconds;child=None;birth=None
    run_root=folder.parent.parent if folder.parent.name=='shared_gpu_admission' else folder.parent
    logs=run_root/'logs';log=None
    def interrupted(signum,frame):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,interrupted)
    try:
        with phase(logs,'queue_for_two_idle_gpus',gpu_ids=pool):
            while True:
                with (ROOT/'.runtime/gpu_admission.lock').open('a') as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX)
                    lease_path=ROOT/'.runtime/gpu_leases.json'
                    leases=read_json(lease_path) if lease_path.exists() else []
                    leases=[l for l in leases if l['start_ticks'] is not None and api.process_ticks(l['pid'])==l['start_ticks']]
                    first=api.snapshot();time.sleep(2);second=api.snapshot()
                    selected=api.select_idle(first,second,{u for l in leases for u in l['uuids']},2,pool)
                    if len(selected)==2:
                        record={'gpus':selected,'before':[first,second],'command':command,'cwd':str(ROOT),
                                'supervisor_pid':os.getpid(),'scope':'one user-authorized two-card training job; independent jobs preserved'}
                        atomic_json(record_path,record)
                        env=child_environment(sys.executable)
                        env.update(CUDA_VISIBLE_DEVICES=','.join(g['uuid'] for g in selected),
                                   STEER_OPD_GPU_ADMISSION=str(record_path))
                        log=(logs/'console.log').open('ab')
                        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                        birth=api.process_ticks(child.pid)
                        record.update(pid=child.pid,start_ticks=birth);atomic_json(record_path,record)
                        leases.append({'pid':child.pid,'start_ticks':birth,'uuids':[g['uuid'] for g in selected]})
                        atomic_json(lease_path,leases)
                        break
                if time.monotonic()>=deadline:raise TimeoutError('The specified GPU pair stayed busy')
                time.sleep(5)
        event(logs,'admitted',pid_child=child.pid,gpus=selected)
        code=child.wait();event(logs,'child_exit',exit_code=code)
        atomic_json(folder/'exit.json',{'exit_code':code,'pid':child.pid,'start_ticks':birth})
        if code:raise SystemExit(code)
    finally:
        if child is not None:api.stop_owned_group(child,birth)
        if log:log.close()


if __name__=='__main__':main()
