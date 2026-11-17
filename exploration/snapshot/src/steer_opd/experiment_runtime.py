"""Low-overhead experiment progress, provenance and whole-call recovery."""
from contextlib import contextmanager
from functools import lru_cache
from datetime import datetime, timezone
import hashlib
import gzip
import io
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import threading
import time
import uuid
import signal
import shutil
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


def require_artifact_space(directory, required_bytes):
    """Use the destination's quota-aware availability, without changing outputs."""
    directory=Path(directory)
    while not directory.exists():directory=directory.parent
    available=shutil.disk_usage(directory).free
    # Full FP32 probabilities dominate; retain room for metadata and other jobs.
    needed=int(required_bytes*1.02)+1024**3
    if needed>available:
        raise OSError(28,f'Insufficient destination quota: need about {needed/1024**3:.1f} GiB, '
            f'available {available/1024**3:.1f} GiB at {directory}. '
            'Keep committed stages; increase quota or explicitly choose another destination.')
    return {'required_bytes':required_bytes,'available_bytes':available,'reserve_bytes':needed-required_bytes}


def timestamp():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def configure_seed(seed=SEED):
    if seed != SEED:
        raise ValueError('The current experiment fixes seed=42 for every process/request')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    # torch.manual_seed also seeds CUDA lazily; do not initialize CUDA here.
    return seed


def child_environment(python=None):
    env={**os.environ, 'PYTHONHASHSEED':'42', 'PYTHONUNBUFFERED':'1', 'VLLM_WORKER_MULTIPROC_METHOD':'spawn',
            'PYTHONPATH':os.pathsep.join([str(ROOT/'src'),str(ROOT/'.runtime/easysteer-plugin')])}
    prefix=Path(python or sys.executable).parent.parent
    cuda=cuda_toolkit(prefix)
    env['CUDA_HOME']=str(cuda)
    env['PATH']=os.pathsep.join([str(prefix/'bin'),str(cuda/'bin'),env.get('PATH','')])
    return env


@lru_cache(maxsize=8)
def cuda_toolkit(prefix):
    """Select a coherent compiler/header pair matching torch's CUDA build."""
    version=torch.version.cuda
    local=Path(prefix)/f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/nvidia/cu13'
    candidates=([Path(os.environ['STEER_OPD_CUDA_HOME'])] if os.environ.get('STEER_OPD_CUDA_HOME') else
                [Path(f'/usr/local/cuda-{version}'),local])
    for path in candidates:
        if not (path/'bin/nvcc').is_file() or not (path/'include/cuda.h').is_file():continue
        compiler=subprocess.check_output([str(path/'bin/nvcc'),'--version'],text=True)
        match=re.search(r'release (\d+\.\d+)',compiler)
        header=re.search(r'^#define CUDA_VERSION\s+(\d+)',(path/'include/cuda.h').read_text(),re.M)
        if match and header and match[1]==version:
            major,minor=map(int,version.split('.'))
            if int(header[1])==major*1000+minor*10:return path
    raise RuntimeError(f'No coherent CUDA{version} compiler/headers; set STEER_OPD_CUDA_HOME to a matching toolkit')


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
    toolkit=cuda_toolkit(Path(sys.prefix))
    compiler_version=subprocess.check_output([str(toolkit/'bin/nvcc'),'--version'],text=True).strip()
    metadata={'schema':1,'seed':42,'request_seed_policy':'all_requests_42',
        'source_revision':revision,'source_bundle':str(local_bundle.relative_to(ROOT)) if local_bundle.is_relative_to(ROOT) else str(local_bundle),
        'python':sys.version,'executable':sys.executable,
        'versions':{p:importlib.metadata.version(p) for p in ('torch','transformers','numpy','tqdm')},
        'pythonhashseed':os.environ.get('PYTHONHASHSEED'),
        'torch_threads':torch.get_num_threads(), 'float32_matmul_precision':torch.get_float32_matmul_precision(),
        'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
        'cudnn_benchmark':torch.backends.cudnn.benchmark,
        'cuda_build':torch.version.cuda,'job_hash':digest(job), 'vllm_worker_multiproc_method':'spawn',
        'batch_invariant_policy':'Qwen3-4B=true; Qwen3.5 roles=false',
        'cuda_toolkit':str(toolkit),'cuda_compiler_version':compiler_version,
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
    binding=digest({k:metadata[k] for k in ('seed','request_seed_policy','source_revision','backend_source_manifest_sha256','versions','pythonhashseed','torch_threads','float32_matmul_precision','deterministic_algorithms','cudnn_benchmark','cuda_build','cuda_toolkit','cuda_compiler_version','vllm_worker_multiproc_method','batch_invariant_policy')})
    path=output/'reproducibility.json'
    if path.exists() and read_json(path)['binding']!=binding:
        raise ValueError('Code/runtime changed: use a new run rather than mixing cached results')
    if not path.exists(): atomic_json(path,{'binding':binding,**metadata})
    return binding


@contextmanager
def engine_startup_guard(seconds=600):
    """Bound initialization only; generation/scoring keep their normal runtime."""
    import faulthandler
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError('Engine initialization must run on the main thread')
    previous=signal.getsignal(signal.SIGALRM)
    def expired(signum,frame):
        faulthandler.dump_traceback(all_threads=True)
        raise TimeoutError(f'EasySteer initialization exceeded {seconds}s; see startup log')
    signal.signal(signal.SIGALRM,expired)
    signal.setitimer(signal.ITIMER_REAL,seconds)
    try:yield
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,previous)


class Progress:
    """Plain stage messages and file heartbeats; native engine tqdm stays live."""
    def __init__(self, output, stage, total, *, initial=0, unit='request', eta=True, interval=15, show_bar=False):
        self.output=Path(output); self.stage=stage; self.total=total; self.done=initial
        self.initial=initial; self.unit=unit; self.eta=eta; self.interval=interval
        self.show_bar=show_bar;self.bar=None;self.committed=initial
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
        if self.show_bar:
            self.bar=tqdm(total=self.total,initial=self.initial,desc=f'{self.stage} [{self.started_at}]',
                          unit=self.unit,dynamic_ncols=True,mininterval=1,file=sys.stderr,
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
            # Structured details and heartbeats stay in files, leaving the
            # terminal available for stage messages and native engine bars.
            label=fields.get('name',fields.get('phase',self.stage))
            messages={
                'stage_start':'开始',
                'stage_requested':'开始阶段',
                'phase_start':'开始',
                'cache_validation_start':'校验已有缓存',
                'cache_reused':f"复用缓存：{fields.get('committed_units',0)} {self.unit}",
                'adopting_live_stage':'等待本项目已在运行的阶段',
                'recovered_after_interruption':'接续上次中断的阶段',
            }
            message=messages.get(event)
            if event=='phase_end':message=f"阶段结束，耗时 {fields['elapsed_seconds']:.1f}s"
            if event=='stage_end':
                status={'complete':'完成','needs_compute':'缓存未完成，需要计算',
                        'interrupted':'已中断','failed':'失败'}.get(fields['status'],fields['status'])
                message=f"{status}，耗时 {state['elapsed_seconds']:.1f}s"
                if fields.get('error'):message+=f"；{fields['error']}"
            if message is not None:
                print(f"[{record['time']}] [{label}] {message}",flush=True)
    def update(self,n=1,*,tokens=0,key=None,**fields):
        with self.lock:
            self.done+=n;self.committed+=n;self.tokens+=tokens
            if self.bar is not None:self.bar.update(n)
            self.last_status.update(fields)
            if self.bar is not None:self.bar.set_postfix(phase=self.phase,remaining=max(0,self.total-self.done),refresh=False)
            self.event('unit_committed',key=key,completed=self.done,total=self.total)
    def resume(self,n):
        with self.lock:
            self.done=n;self.initial=n;self.committed=n
            if self.bar is not None:self.bar.update(n)
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
            if self.bar is not None:self.bar.update(max(0,completed-self.done))
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
                if self.bar is not None:
                    self.bar.set_postfix(phase=self.phase,remaining=max(0,self.total-self.done),refresh=False)
                    self.bar.refresh()
                self.event('heartbeat',**self._state('running'))
    def __exit__(self,kind,value,tb):
        self.stop.set();self.thread.join();self.closed=True
        self.event('stage_end',status='complete' if kind is None else 'needs_compute' if isinstance(value,SystemExit) and value.code==75 else 'interrupted' if issubclass(kind,(KeyboardInterrupt,SystemExit)) else 'failed',
                   finished_at=timestamp(),error=None if value is None else repr(value))
        if self.bar is not None:self.bar.close()
        self.lease.close()
        if self.previous_sigterm is not None:signal.signal(signal.SIGTERM,self.previous_sigterm)


class StageCheckpoint:
    """One manifest commits a complete engine call, never individual requests."""
    schema = "engine_call_v1"

    def __init__(self, output, binding, *, tensor_compression=None):
        self.output = Path(output)
        self.path = self.output / "complete.json"
        self.binding = binding
        identity = self.output / "stage_input.json"
        saved = read_json(identity) if identity.exists() else None
        self.tensor_compression = (saved.get('tensor_compression','none') if saved and tensor_compression is None
                                   else tensor_compression or 'none')
        if self.tensor_compression not in ('none','gzip'):raise ValueError('Unknown lossless tensor compression')
        expected = {"binding": binding, "schema": self.schema}
        if self.tensor_compression!='none':expected['tensor_compression']=self.tensor_compression
        if saved is not None and saved != expected:
            raise ValueError("Stage input identity changed")
        if saved is None:atomic_json(identity,expected)

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

    def compact_hidden_states(self):
        """Retire only old per-token SV states after validating their whole commit.

        Keep the original manifest in a small receipt. Atomically publish the
        retained-file manifest before unlinking states, so interrupted cleanup
        can resume without weakening normal missing-file/integrity checks.
        Pool bytes, stage binding, banks and downstream results stay unchanged.
        """
        value = self.load()
        if value is None or 'pooled.pt' not in value['files']:
            raise ValueError('A committed SV activation pool is required before removing hidden states')
        receipt_path = self.output/'hidden_state_cleanup.json'
        compacted = value.get('hidden_state_storage') == 'pooled_only'
        if compacted:
            if value.get('retention_receipt') != receipt_path.name:
                return {'removed_files':0,'freed_bytes':0}
            receipt = read_json(receipt_path)
            if (receipt.get('schema') != 'sv_hidden_state_cleanup_v1'
                    or receipt['original_complete']['binding'] != self.binding):
                raise ValueError('Hidden-state cleanup receipt changed')
            original = receipt['original_complete']
        else:
            original = value
        raw = {name:item for name,item in original['files'].items() if name.startswith('token_activations/')}
        for name in raw:
            parts = Path(name).parts
            if len(parts) != 3 or parts[1] in ('.','..') or parts[2] not in ('teacher.pt','student.pt'):
                raise ValueError('Cleanup is restricted to teacher/self per-token hidden-state files')
        if not raw:
            return {'removed_files':0,'freed_bytes':0}
        if compacted:
            for name,item in raw.items():
                path = self.output/name
                if path.exists() and file_sha256(path) != item['sha256']:
                    raise ValueError('Remaining hidden-state file changed during cleanup')
        else:
            receipt = {'schema':'sv_hidden_state_cleanup_v1','created_at':timestamp(),
                'reason':'User requested no per-token hidden-state persistence; preserve pooled means and SV banks',
                'original_complete_sha256':file_sha256(self.path),'original_complete':original,
                'retired_files':len(raw),'retired_bytes':sum(x['bytes'] for x in raw.values()),
                'legacy_pool_token_paths':'Historical provenance only; raw states intentionally retired'}
            atomic_json(receipt_path,receipt)
            retained = {k:v for k,v in value['files'].items() if k not in raw}
            stat = receipt_path.stat()
            retained[receipt_path.name] = {'sha256':file_sha256(receipt_path),'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}
            atomic_json(self.path,{**value,'files':retained,'hidden_state_storage':'pooled_only',
                'retention_receipt':receipt_path.name,'compacted_at':timestamp(),
                'original_complete_sha256':receipt['original_complete_sha256']})
        removed = freed = 0
        for name in raw:
            path = self.output/name
            if path.exists():
                size = path.stat().st_size
                path.unlink(); removed += 1; freed += size
        folder = self.output/'token_activations'
        if folder.exists():
            for path in sorted((p for p in folder.rglob('*') if p.is_dir()),key=lambda p:len(p.parts),reverse=True):
                if not any(path.iterdir()):path.rmdir()
            if not any(folder.iterdir()):folder.rmdir()
        self.load()
        return {'removed_files':removed,'freed_bytes':freed}

    def load_tensors(self, keys):
        value = self.load()
        if value is None:
            return None
        if set(value["record_keys"]) != set(keys):
            raise ValueError("Committed stage response keys changed")
        compression=value.get('tensor_compression','none')
        if compression not in ('none','gzip'):raise ValueError('Unknown committed tensor compression')
        result={}
        for key in keys:
            if compression=='gzip':
                source=io.BytesIO(gzip.decompress((self.output/f'{key}.pt.gz').read_bytes()))
            else:source=self.output/f'{key}.pt'
            result[key]=torch.load(source,map_location='cpu',weights_only=True)
        return result

    def save_tensors(self, records, extra_files=()):
        if self.path.exists():
            raise ValueError("A committed stage cannot be overwritten")
        files = list(extra_files)
        for key, record in records.items():
            if Path(key).name != key or key in (".", ".."):
                raise ValueError("Unsafe response key")
            if self.tensor_compression=='gzip':
                path=self.output/f'{key}.pt.gz';path.parent.mkdir(parents=True,exist_ok=True)
                buffer=io.BytesIO();torch.save(record,buffer)
                packed=gzip.compress(buffer.getvalue(),compresslevel=6,mtime=0)
                tmp=path.with_name(path.name+f'.tmp.{os.getpid()}')
                with tmp.open('wb') as stream:
                    stream.write(packed);stream.flush();os.fsync(stream.fileno())
                os.replace(tmp,path)
            else:
                path = self.output / f"{key}.pt"
                atomic_torch(path, record)
            files.append(path)
        metadata={'record_keys':list(records)}
        if self.tensor_compression!='none':metadata['tensor_compression']=self.tensor_compression
        self.commit(files,**metadata)


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
    run=Path(run)
    if cfg.get('layout_version')==1:
        from steer_opd.artifact_reuse import StageArtifacts
        from steer_opd.stage_io import project
        sources={};unique={}
        cache=StageArtifacts(cfg)
        for ref in sorted((run/'metadata/upstreams').glob('*.json')):
            value=read_json(ref);descriptor=project(value['descriptor'])
            if file_sha256(descriptor)!=value['descriptor_sha256']:
                raise ValueError('Upstream descriptor changed before export')
            owner,packet=cache.validate(descriptor)
            if packet['artifact_id']!=value['artifact_id']:raise ValueError('Upstream identity changed')
            sources[ref.stem]={**value,'files':packet['files']}
            for name,item in packet['files'].items():unique[str(owner/name)]=item['bytes']
        atomic_json(run/'metadata/artifact_index.json',{'schema':2,'step_dirs':cfg['step_dirs'],
            'sources':sources,'unique_bytes':sum(unique.values()),'parameter_updates':0,
            'created_at':timestamp(),'seed':42,'large_files_copied':False,
            'reuse_scope':'Exact checkpoint, source prefixes, tokenizer, protocol and committed SHA identities',
            'training_boundary':'New trajectories/checkpoints require new matching probabilities and activations.'})
        return
    known={}
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
    for name in ('references','source','source_overlap','activations','activations_teacher','banks','banks_teacher','alignment','native_scores','select','confirm','provenance'):
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
    for path in [run/'config.json',run/'strength_schedule.json',Path(ROOT/cfg['data_dir'])/'questions.json',Path(ROOT/cfg['data_dir'])/'split.json']:
        if path.exists(): files[str(path.resolve())]={'sha256':file_sha256(path),'bytes':path.stat().st_size}
    atomic_json(run/'artifact_index.json',{'schema':1,'files':files,'unique_bytes':unique_bytes,
        'experiment':cfg.get('experiment'),'phase':cfg.get('phase'),
        'parameter_updates':0,'created_at':timestamp(),'seed':42,
        'source_policy':'unsteered_S0; candidate scores use those same fixed source prefixes',
        'reuse_scope':'reference texts, pooled activation means/norms, banks, masks, raw Top16 probabilities and observed-span scores; no per-token hidden states',
        'training_boundary':'Not a trained checkpoint or new controlled rollout. New prefixes/checkpoints need their own behavior and teacher scores.'})
