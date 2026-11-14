"""Low-overhead experiment progress, provenance and whole-call recovery."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import threading
import time
import uuid
import signal
from zoneinfo import ZoneInfo

import numpy as np
import torch
from tqdm import tqdm

from steer_opd.resumable_core import ROOT, atomic_json, atomic_torch, digest, read_json
from steer_opd.direction_bank import file_sha256

SEED = 42


def compute_required():
    if os.environ.get('STEER_OPD_CACHE_ONLY') == '1':
        raise SystemExit(75)


def timestamp():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def configure_seed(seed=SEED):
    if seed != SEED:
        raise ValueError('The current experiment fixes seed=42 for every process/request')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    # torch.manual_seed also seeds CUDA lazily; do not initialize CUDA here.
    return seed


def child_environment(python=None):
    env={**os.environ, 'PYTHONHASHSEED':'42', 'PYTHONUNBUFFERED':'1',
            'PYTHONPATH':os.pathsep.join([str(ROOT/'src'),str(ROOT/'.runtime/easysteer-plugin')])}
    prefix=Path(python or sys.executable).parent.parent
    cuda=prefix/f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/nvidia/cu13'
    if (cuda/'bin/nvcc').is_file():
        env['CUDA_HOME']=str(cuda)
        env['PATH']=os.pathsep.join([str(prefix/'bin'),str(cuda/'bin'),env.get('PATH','')])
    return env


def source_revision():
    files=sorted([*(ROOT/'src/steer_opd').glob('*.py'),*(ROOT/'scripts').glob('*.py')])
    bodies={str(p.relative_to(ROOT)):p.read_text() for p in files}
    hashes={p:hashlib.sha256(s.encode()).hexdigest() for p,s in bodies.items()}
    return digest(hashes), hashes, bodies


def initialize_stage(job, output):
    configure_seed(job.get('seed',42))
    output=Path(output); revision, hashes, bodies=source_revision()
    # Once per revision, retain the exact code; avoid a large git diff/history dump.
    bundle=ROOT/'.runtime/source_revisions'/revision/'source_bundle.json'
    if not bundle.exists(): atomic_json(bundle,{'sha256':hashes,'source':bodies})
    run_root=Path(os.environ.get('STEER_OPD_RUN_ROOT',output))
    if not run_root.resolve().is_relative_to(ROOT.resolve()): run_root=output
    local_bundle=run_root/'provenance'/revision/'source_bundle.json'
    if not local_bundle.exists():
        local_bundle.parent.mkdir(parents=True,exist_ok=True)
        os.link(bundle,local_bundle)
    metadata={'schema':1,'seed':42,'request_seed_policy':'all_requests_42',
        'source_revision':revision,'source_bundle':str(local_bundle.relative_to(ROOT)) if local_bundle.is_relative_to(ROOT) else str(local_bundle),
        'python':sys.version,'executable':sys.executable,
        'versions':{p:importlib.metadata.version(p) for p in ('torch','transformers','numpy','tqdm')},
        'pythonhashseed':os.environ.get('PYTHONHASHSEED'),
        'torch_threads':torch.get_num_threads(), 'float32_matmul_precision':torch.get_float32_matmul_precision(),
        'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
        'cudnn_benchmark':torch.backends.cudnn.benchmark,
        'cuda_build':torch.version.cuda,'job_hash':digest(job),
        'resume_semantics':'complete engine calls reused; unfinished calls restart from seed42; no in-engine checkpoints'}
    for p in ('vllm','easysteer','triton','flashinfer-python','tokenizers','safetensors','nvidia-cudnn-cu13','nvidia-nccl-cu13','nvidia-cuda-runtime'):
        try: metadata['versions'][p]=importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError: pass
    inventory={d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata.get('Name')}
    environment_file=local_bundle.parent/('environment_'+digest(inventory)+'.json')
    if not environment_file.exists():atomic_json(environment_file,inventory)
    metadata['environment_inventory']=str(environment_file)
    backend_manifest=ROOT/'ref/easysteer_runtime/manifest.json'
    metadata['backend_source_manifest_sha256']=file_sha256(backend_manifest)
    if 'easysteer' in metadata['versions']:
        dist=importlib.metadata.distribution('vllm')
        for item in read_json(backend_manifest)['files']:
            rel=Path(item['path']).relative_to('ref/easysteer_runtime')
            if rel.parts[0]=='vllm' and file_sha256(dist.locate_file(rel))!=item['sha256']:
                raise ValueError(f'Installed EasySteer fork source changed: {rel}')
    binding=digest({k:metadata[k] for k in ('seed','request_seed_policy','source_revision','backend_source_manifest_sha256','versions','pythonhashseed','torch_threads','float32_matmul_precision','deterministic_algorithms','cudnn_benchmark','cuda_build')})
    path=output/'reproducibility.json'
    if path.exists() and read_json(path)['binding']!=binding:
        raise ValueError('Code/runtime changed: use a new run rather than mixing cached results')
    if not path.exists(): atomic_json(path,{'binding':binding,**metadata})
    return binding


class Progress:
    """One heartbeat thread writes timestamps/ETA even inside a blocking engine call."""
    def __init__(self, output, stage, total, *, initial=0, unit='request', eta=True, interval=15, show_bar=True):
        self.output=Path(output); self.stage=stage; self.total=total; self.done=initial
        self.initial=initial; self.unit=unit; self.eta=eta; self.interval=interval
        self.show_bar=show_bar;self.committed=initial
        self.attempt_id=f'{time.time_ns()}_{os.getpid()}_{uuid.uuid4().hex[:6]}'
        self.attempt=self.output/'attempts'/self.attempt_id
        self.lock=threading.RLock(); self.stop=threading.Event()
        self.started=time.monotonic(); self.started_at=timestamp(); self.phase='initializing'
        self.tokens=0; self.last_status={}; self.closed=False
        previous=read_json(self.output/'progress.json') if (self.output/'progress.json').exists() else {}
        self.prior_active=float(previous.get('active_elapsed_seconds',0))
        self.first_started_at=previous.get('first_started_at',self.started_at)
        self.previous=previous
    def __enter__(self):
        self.output.mkdir(parents=True,exist_ok=True)
        self.lease=(self.output/'.stage.lock').open('a')
        try: fcntl.flock(self.lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            self.lease.close();raise RuntimeError(f'Stage already has a live writer: {self.output}')
        self.attempt.mkdir(parents=True,exist_ok=True)
        self.bar=tqdm(total=self.total,initial=self.initial,desc=f'{self.stage} [{self.started_at}]',
                      unit=self.unit,dynamic_ncols=True,mininterval=1,file=sys.stderr,disable=not self.show_bar,
                      bar_format=None if self.eta else '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}, ETA ?, {rate_fmt}{postfix}]')
        self.previous_sigterm=None
        if threading.current_thread() is threading.main_thread():
            self.previous_sigterm=signal.getsignal(signal.SIGTERM)
            def interrupted(signum,frame): raise KeyboardInterrupt(f'signal {signum}')
            signal.signal(signal.SIGTERM,interrupted)
        self.event('stage_start',resumed_units=self.initial)
        if self.previous.get('status')=='running':
            self.event('recovered_after_interruption',previous_attempt=self.previous.get('attempt_id'),last_heartbeat=self.previous.get('updated_at'))
        self.thread=threading.Thread(target=self._heartbeat,daemon=True); self.thread.start()
        return self
    def _state(self,status):
        elapsed=time.monotonic()-self.started; fresh=self.done-self.initial
        eta=(elapsed/fresh*(self.total-self.done) if self.eta and fresh>0 else None)
        return {'stage':self.stage,'attempt_id':self.attempt_id,'pid':os.getpid(),'status':status,
            'first_started_at':self.first_started_at,'started_at':self.started_at,'updated_at':timestamp(),
            'elapsed_seconds':elapsed,'active_elapsed_seconds':self.prior_active+elapsed,
            'wall_since_first_start_seconds':(datetime.now(timezone.utc)-datetime.fromisoformat(self.first_started_at)).total_seconds(),
            'completed':self.done,'committed':self.committed,'resumed':self.initial,'total':self.total,'unit':self.unit,
            'eta_seconds':eta,'eta_basis':'completed_requests_this_attempt' if eta is not None else 'unavailable',
            'phase':self.phase,'completed_tokens_this_attempt':self.tokens,
            'completed_token_per_second':self.tokens/elapsed if elapsed else 0,**self.last_status}
    def event(self,event,**fields):
        with self.lock:
            record={'event':event,'time':timestamp(),'stage':self.stage,'attempt_id':self.attempt_id,**fields}
            self.output.mkdir(parents=True,exist_ok=True)
            with (self.output/'events.jsonl').open('a') as f:
                f.write(json.dumps(record,ensure_ascii=False,default=str)+'\n');f.flush()
            state=self._state('running' if not self.closed else fields.get('status','complete'))
            atomic_json(self.output/'progress.json',state)
            atomic_json(self.attempt/'progress.json',state)
            # Keep plain timestamped lines in stdout.log as well as tqdm on stderr.
            if event not in ('unit_committed',): print(json.dumps(record,ensure_ascii=False,default=str),flush=True)
    def update(self,n=1,*,tokens=0,key=None,**fields):
        with self.lock:
            self.done+=n;self.committed+=n;self.tokens+=tokens;self.bar.update(n)
            self.last_status.update(fields)
            self.bar.set_postfix(phase=self.phase,remaining=max(0,self.total-self.done),refresh=False)
            self.event('unit_committed',key=key,completed=self.done,total=self.total)
    def resume(self,n):
        with self.lock:
            self.done=n;self.initial=n;self.committed=n;self.bar.update(n)
            self.event('cache_reused',committed_units=n)
    def engine_tqdm(self, *args, **kwargs):
        """Native engine progress updates counters only; heartbeat handles files."""
        outer = self
        track = kwargs.get('desc') == 'Processed prompts'
        kwargs.update(mininterval=1, dynamic_ncols=True)
        class EngineBar(tqdm):
            def update(self, n=1):
                result = super().update(n)
                if track:
                    with outer.lock:
                        outer.done += n
                return result
        return EngineBar(*args, **kwargs)

    def finished_call(self, completed):
        with self.lock:
            if self.show_bar:self.bar.update(max(0,completed-self.done))
            self.done = completed
            self.event('engine_call_finished', completed=completed)

    def mark_committed(self, n):
        with self.lock:
            self.committed = n
            self.event('stage_checkpoint_committed', committed=n)

    @contextmanager
    def phase_scope(self,name,**fields):
        before=self.phase;self.phase=name;started=time.monotonic()
        self.event('phase_start',phase=name,**fields)
        try: yield
        finally:
            self.event('phase_end',phase=name,elapsed_seconds=time.monotonic()-started)
            self.phase=before
    def _heartbeat(self):
        while not self.stop.wait(self.interval):
            with self.lock:
                self.bar.set_postfix(phase=self.phase,remaining=max(0,self.total-self.done),refresh=False)
                self.bar.refresh()
                self.event('heartbeat',**self._state('running'))
    def __exit__(self,kind,value,tb):
        self.stop.set();self.thread.join();self.closed=True
        self.event('stage_end',status='complete' if kind is None else 'needs_compute' if isinstance(value,SystemExit) and value.code==75 else 'interrupted' if issubclass(kind,(KeyboardInterrupt,SystemExit)) else 'failed',
                   finished_at=timestamp(),error=None if value is None else repr(value))
        self.bar.close()
        self.lease.close()
        if self.previous_sigterm is not None:signal.signal(signal.SIGTERM,self.previous_sigterm)


class StageCheckpoint:
    """One manifest commits a complete engine call, never individual requests."""
    schema = "engine_call_v1"

    def __init__(self, output, binding):
        self.output = Path(output)
        self.path = self.output / "complete.json"
        self.binding = binding
        identity = self.output / "stage_input.json"
        if identity.exists() and read_json(identity) != {"binding": binding, "schema": self.schema}:
            raise ValueError("Stage input identity changed")
        if not identity.exists():
            atomic_json(identity, {"binding": binding, "schema": self.schema})

    def load(self):
        if not self.path.exists():
            return None
        value = read_json(self.path)
        if value.get("schema") != self.schema or value.get("binding") != self.binding:
            raise ValueError("Stage checkpoint binding changed")
        for name, item in value["files"].items():
            path = self.output / name
            if not path.resolve().is_relative_to(self.output.resolve()):
                raise ValueError("Checkpoint path escape")
            if not path.is_file() or file_sha256(path) != item["sha256"]:
                raise ValueError(f"Committed stage file missing or changed: {name}")
        return value

    def commit(self, files, **metadata):
        if self.path.exists():
            raise ValueError("A committed stage cannot be overwritten")
        entries = {}
        for file in files:
            path = Path(file)
            name = str(path.relative_to(self.output))
            stat = path.stat()
            entries[name] = {"sha256": file_sha256(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        atomic_json(self.path, {"schema": self.schema, "binding": self.binding,
            "files": entries, "committed_at": timestamp(), "parameter_updates": 0, **metadata})

    def load_tensors(self, keys):
        value = self.load()
        if value is None:
            return None
        if set(value["record_keys"]) != set(keys):
            raise ValueError("Committed stage response keys changed")
        return {key: torch.load(self.output / f"{key}.pt", map_location="cpu", weights_only=True) for key in keys}

    def save_tensors(self, records, extra_files=()):
        if self.path.exists():
            raise ValueError("A committed stage cannot be overwritten")
        files = list(extra_files)
        for key, record in records.items():
            if Path(key).name != key or key in (".", ".."):
                raise ValueError("Unsafe response key")
            path = self.output / f"{key}.pt"
            atomic_torch(path, record)
            files.append(path)
        self.commit(files, record_keys=list(records))


@contextmanager
def console_log(path):
    """Append a full console transcript while keeping stdout/stderr live."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    lock=threading.RLock();original=(sys.stdout,sys.stderr)
    with path.open('a',encoding='utf-8',newline='') as log:
        class Tee:
            def __init__(self,stream): self.stream=stream
            def write(self,text):
                with lock:
                    log.write(text);log.flush()
                    try: return self.stream.write(text)
                    except (BrokenPipeError,OSError): return len(text)
            def flush(self):
                with lock:
                    log.flush()
                    try:self.stream.flush()
                    except (BrokenPipeError,OSError):pass
            def __getattr__(self,name): return getattr(self.stream,name)
        sys.stdout,sys.stderr=Tee(original[0]),Tee(original[1])
        try:yield
        finally:sys.stdout,sys.stderr=original


def export_artifacts(cfg,run):
    """Archive original prefixes/likelihoods and their meaning for future training."""
    run=Path(run);known={}
    for marker in run.rglob('complete.json'):
        value = read_json(marker)
        if value.get('schema') != StageCheckpoint.schema:
            continue
        for name, item in value['files'].items():
            path = marker.parent / name
            stat = path.stat()
            if stat.st_size != item['bytes'] or stat.st_mtime_ns != item['mtime_ns']:
                raise ValueError('Committed artifact metadata changed before export')
            known[path.resolve()] = item['sha256']
    files={};seen={};unique_bytes=0
    for name in ('references','source','banks','alignment','native_scores','select','confirm','provenance'):
        folder=run/name
        if not folder.exists():continue
        for path in sorted(folder.rglob('*')):
            if not path.is_file() or any(x in path.parts for x in ('attempts','capture_progress','bank_progress')): continue
            if path.name.startswith('.') or path.name in ('events.jsonl','progress.json'):continue
            if '.tmp.' in path.name or '.link.' in path.name:continue
            stat=path.stat();inode=(stat.st_dev,stat.st_ino)
            sha=known.get(path.resolve()) or seen.get(inode)
            if sha is None:sha=file_sha256(path)
            if inode not in seen:unique_bytes+=stat.st_size
            seen[inode]=sha
            files[str(path.relative_to(run))]={'sha256':sha,'bytes':stat.st_size}
    for path in [run/'config.json',Path(ROOT/cfg['data_dir'])/'questions.json',Path(ROOT/cfg['data_dir'])/'split.json']:
        if path.exists(): files[str(path.resolve())]={'sha256':file_sha256(path),'bytes':path.stat().st_size}
    atomic_json(run/'artifact_index.json',{'schema':1,'files':files,'unique_bytes':unique_bytes,
        'parameter_updates':0,'created_at':timestamp(),'seed':42,
        'source_policy':'unsteered_S0; candidate scores use those same fixed source prefixes',
        'reuse_scope':'reference texts, BF16 activations, banks, masks, full native probability summaries and observed-span scores',
        'training_boundary':'Not a trained checkpoint or new controlled rollout. New prefixes/checkpoints need their own behavior and teacher scores.'})
