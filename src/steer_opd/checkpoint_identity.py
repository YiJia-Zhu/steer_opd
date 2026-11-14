"""Content-bound frozen checkpoints; one streaming hash per unchanged shard."""
import hashlib
from pathlib import Path
from time import time_ns

from steer_opd.resumable_core import ROOT, atomic_json, digest, read_json


def _signature(stat):
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def checkpoint_content_id(model_path, cache_dir=None, receipt_dir=None):
    model = Path(model_path).resolve()
    index = model / 'model.safetensors.index.json'
    if index.is_file():
        weights = sorted(set(read_json(index)['weight_map'].values()))
    elif (model / 'model.safetensors').is_file():
        weights = ['model.safetensors']
    else:
        raise FileNotFoundError(f'Complete checkpoint weights are required: {model}')
    files = sorted(set(weights) | {p.name for p in model.iterdir() if p.name in
        ('config.json','generation_config.json','tokenizer.json','tokenizer_config.json','model.safetensors.index.json',
         'chat_template.jinja','special_tokens_map.json','added_tokens.json','vocab.json','merges.txt')})
    cache_dir = ROOT / '.runtime/checkpoint_fingerprints' if cache_dir is None else Path(cache_dir)
    cache_path = cache_dir / (digest(str(model))+'.json')
    cached = read_json(cache_path) if cache_path.exists() else {}
    old = cached.get('files', {})
    records = {}
    for name in files:
        path = model / name
        stat = path.stat()
        signature = _signature(stat)
        value = old.get(name, {})
        # Some filesystems timestamp rapid same-size writes within one clock
        # tick identically. Never reuse hashes while metadata is this recent.
        settled = max(stat.st_mtime_ns, stat.st_ctime_ns) < time_ns()-2_000_000_000
        if value.get('signature') != signature or not settled:
            sha = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(8*1024**2), b''):
                    sha.update(block)
            if _signature(path.stat()) != signature:
                raise RuntimeError('Checkpoint changed while computing its content identity')
            value = {'signature': signature, 'sha256': sha.hexdigest()}
        records[name] = value
    identity = digest({name: value['sha256'] for name, value in records.items()})
    payload = {'schema':1, 'files':records, 'content_id':identity}
    if cached != payload:
        atomic_json(cache_path, payload)
    if receipt_dir is not None:
        receipt=Path(receipt_dir)/(identity+'.json')
        if not receipt.exists(): atomic_json(receipt,payload)
    return identity
