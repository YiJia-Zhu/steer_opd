"""Reuse the project's PID-birth/session-checked cleanup for training subprocesses."""
import importlib.util
from pathlib import Path
import subprocess
import time

from steer_opd.resumable_core import ROOT


def cleanup_api():
    spec=importlib.util.spec_from_file_location('steer_training_idle_gpu_runtime',ROOT/'scripts/run_on_idle_gpus.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class ProcessHandle:
    def __init__(self,pid,start_ticks):self.pid=pid;self.start_ticks=start_ticks
    def wait(self,timeout=None):
        api=cleanup_api();started=time.monotonic()
        while api.process_ticks(self.pid)==self.start_ticks:
            if timeout is not None and time.monotonic()-started>timeout:
                raise subprocess.TimeoutExpired(str(self.pid),timeout)
            time.sleep(.05)
        return 0


def stop_session(pid,start_ticks,*,child=None,grace=10):
    cleanup_api().stop_owned_group(child or ProcessHandle(pid,start_ticks),start_ticks,grace=grace)
