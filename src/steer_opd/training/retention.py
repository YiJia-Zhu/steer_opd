"""Bound transient batch storage and rotate only fully committed checkpoints."""
from pathlib import Path
import os
import re
import shutil
import uuid

from steer_opd.direction_bank import file_sha256
from steer_opd.resumable_core import atomic_json, digest, read_json
from steer_opd.retention import MINIMAL, minimal, ALL_MODELS_LATEST_STATE, keep_model_history

MODEL_RECEIPT='model_checkpoint.complete.json'
STATE_RETIREMENT='state_retirement.json'
CHECKPOINT_METADATA={'checkpoint.complete.json',MODEL_RECEIPT,STATE_RETIREMENT}


def _resume_only(name):
    return name=='data.pt' or re.fullmatch(r'actor/(?:optim|extra_state)_world_size_[0-9]+_rank_[0-9]+\.pt',name) is not None


def batch_root(folder, plan):
    return Path(folder) / 'batch' if minimal(plan) else Path(folder)


def _owner(plan):
    return {'schema': 'opd_batch_workspace_v1', 'retention_policy': MINIMAL,
            'plan_binding': plan['binding']}


def reset_batch_workspace(folder, plan):
    """Call only with the client lock held and all scorers asleep/stopped.

    Failed batch contents survive until retry/restart. Once an update succeeds,
    atomically detach its workspace before deleting it. A crash during deletion
    leaves a recognized, owned garbage directory that the next reset finishes.
    """
    if not minimal(plan):return {'files': 0, 'bytes': 0}
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    workspace = batch_root(folder, plan)
    candidates = list(folder.glob('.batch_gc_*'))
    if workspace.exists() or workspace.is_symlink():candidates.append(workspace)
    # Validate every candidate before any mutation. Never follow external links.
    for path in candidates:
        if path.is_symlink() or not path.is_dir():raise ValueError('Unsafe batch workspace')
        if not (path/'owner.json').exists() and not any(path.iterdir()):continue
        if read_json(path / 'owner.json') != _owner(plan):
            raise ValueError('Batch workspace belongs to another plan')
        for item in path.rglob('*'):
            if item.is_symlink():raise ValueError('Symlink inside batch workspace')
    stats = {'files': 0, 'bytes': 0}
    if workspace.exists():
        garbage = folder / ('.batch_gc_' + uuid.uuid4().hex)
        workspace.rename(garbage)
        candidates = [garbage if p == workspace else p for p in candidates]
    workspace.mkdir()
    atomic_json(workspace / 'owner.json', _owner(plan))
    for path in candidates:
        for item in path.rglob('*'):
            if item.is_file():
                stats['files'] += 1
                stats['bytes'] += item.stat().st_size
        shutil.rmtree(path)
    return stats


def _checkpoint_entries(folder,*,sync=False,model_only=False):
    if folder.is_symlink():raise ValueError('Symlinked training checkpoint')
    files = {};directories={folder}
    for path in sorted(folder.rglob('*')):
        if path.is_symlink():raise ValueError('Symlink inside training checkpoint')
        name=str(path.relative_to(folder))
        if path.is_file() and name not in CHECKPOINT_METADATA and not (model_only and _resume_only(name)):
            if sync:
                with path.open('rb') as stream:os.fsync(stream.fileno())
                parent=path.parent
                while parent!=folder:
                    directories.add(parent);parent=parent.parent
            files[name] = {'bytes': path.stat().st_size,
                                                   'sha256': file_sha256(path)}
    if sync:
        for directory in sorted(directories,key=lambda p:len(p.parts),reverse=True):_sync_directory(directory)
    return files


def validate_checkpoint(folder, plan):
    folder = Path(folder)
    receipt = read_json(folder / 'checkpoint.complete.json')
    if (receipt.get('schema') != 'opd_checkpoint_v1' or receipt.get('plan_binding') != plan['binding']
            or receipt.get('binding') != digest({k: v for k, v in receipt.items() if k != 'binding'})
            or receipt.get('files') != _checkpoint_entries(folder)):
        raise ValueError('Training checkpoint identity or files changed')
    return receipt


def _read_full_receipt(folder,plan):
    saved=read_json(folder/'checkpoint.complete.json')
    if (folder.is_symlink() or saved.get('schema')!='opd_checkpoint_v1'
            or saved.get('plan_binding')!=plan['binding']
            or saved.get('binding')!=digest({k:v for k,v in saved.items() if k!='binding'})):
        raise ValueError('Checkpoint receipt identity changed')
    return saved


def _model_receipt(folder,full,plan):
    return {'schema':'opd_model_checkpoint_v1','plan_binding':plan['binding'],'step':full['step'],
        'checkpoint_retention':ALL_MODELS_LATEST_STATE,
        'source_receipt_sha256':file_sha256(folder/'checkpoint.complete.json'),
        'files':{name:item for name,item in full['files'].items() if not _resume_only(name)}}


def _commit_model_receipt(folder,full,plan):
    packet=_model_receipt(folder,full,plan)
    packet['binding']=digest(packet)
    atomic_json(folder/MODEL_RECEIPT,packet)
    _sync_directory(folder)
    return packet


def validate_model_checkpoint(folder,plan):
    """Validate retained model shards/config without requiring retired optimizer state."""
    folder=Path(folder);full=_read_full_receipt(folder,plan)
    packet=read_json(folder/MODEL_RECEIPT)
    expected=_model_receipt(folder,full,plan)
    expected['binding']=digest(expected)
    ranks=len(plan['settings']['gpu_pool'])
    required={f'actor/model_world_size_{ranks}_rank_{rank}.pt' for rank in range(ranks)}
    if (full.get('checkpoint_retention')!=ALL_MODELS_LATEST_STATE or packet!=expected
            or not required.issubset(packet['files'])
            or packet['files']!=_checkpoint_entries(folder,model_only=True)):
        raise ValueError('Retained model checkpoint identity or files changed')
    if (folder/STATE_RETIREMENT).exists():_retirement_receipt(folder,plan,packet)
    return packet


def _retirement_receipt(folder,plan,model):
    full=_read_full_receipt(folder,plan)
    removed={name:item for name,item in full['files'].items() if _resume_only(name)}
    packet=read_json(folder/STATE_RETIREMENT)
    if (packet.get('schema')!='opd_checkpoint_state_retirement_v1'
            or packet.get('plan_binding')!=plan['binding'] or packet.get('step')!=model['step']
            or packet.get('model_receipt_sha256')!=file_sha256(folder/MODEL_RECEIPT)
            or packet.get('source_receipt_sha256')!=model['source_receipt_sha256']
            or packet.get('removed_files')!=removed
            or packet.get('binding')!=digest({k:v for k,v in packet.items() if k!='binding'})):
        raise ValueError('Checkpoint state retirement identity changed')
    return packet


def retire_checkpoint_state(folder,plan,*,replacement):
    """Commit evidence first, then remove only known resume files; retry is idempotent.

    The caller must first validate and publish the newer complete checkpoint.
    All retained model bytes and all still-present state files are checked before
    the first unlink. A failed cleanup leaves the newer restore point intact.
    """
    folder=Path(folder)
    if not keep_model_history(plan):raise ValueError('Model history retention is required')
    model=validate_model_checkpoint(folder,plan)
    full=_read_full_receipt(folder,plan)
    removed={name:item for name,item in full['files'].items() if _resume_only(name)}
    marker=folder/STATE_RETIREMENT
    if marker.exists():packet=_retirement_receipt(folder,plan,model)
    else:
        packet={'schema':'opd_checkpoint_state_retirement_v1','plan_binding':plan['binding'],
            'step':model['step'],'source_receipt_sha256':model['source_receipt_sha256'],
            'model_receipt_sha256':file_sha256(folder/MODEL_RECEIPT),'removed_files':removed,
            'replacement':replacement}
        packet['binding']=digest(packet)
    # Missing files are legitimate only after the durable retirement marker.
    for name,item in removed.items():
        path=folder/name
        if path.is_symlink():raise ValueError('Symlink inside checkpoint state')
        if not path.exists():
            if marker.exists():continue
            raise ValueError('Unexplained missing checkpoint state before retirement')
        if path.stat().st_size!=item['bytes'] or file_sha256(path)!=item['sha256']:
            raise ValueError('Checkpoint state changed before retirement')
    if not marker.exists():
        atomic_json(marker,packet);_sync_directory(folder)
    for name in removed:(folder/name).unlink(missing_ok=True)
    for parent in {folder,(folder/'actor')}:
        if parent.is_dir():_sync_directory(parent)
    return packet


def commit_checkpoint(root, step, plan):
    """Called after upstream's synchronous all-rank save and dataloader save."""
    root = Path(root)
    folder = root / f'global_step_{step}'
    tracker = root / 'latest_checkpointed_iteration.txt'
    if tracker.read_text().strip() != str(step):
        raise ValueError('Upstream did not commit the requested checkpoint step')
    files = _checkpoint_entries(folder,sync=True)
    required = {'data.pt'}
    for rank in range(len(plan['settings']['gpu_pool'])):
        for kind in ('model', 'optim', 'extra_state'):
            required.add(f'actor/{kind}_world_size_{len(plan["settings"]["gpu_pool"])}_rank_{rank}.pt')
    if not required.issubset(files) or any(files[name]['bytes'] <= 0 for name in required):
        raise ValueError('Incomplete model/optimizer/RNG/dataloader checkpoint')
    receipt = {'schema': 'opd_checkpoint_v1', 'retention_policy': MINIMAL,
               'plan_binding': plan['binding'], 'step': step, 'files': files}
    if keep_model_history(plan):receipt['checkpoint_retention']=ALL_MODELS_LATEST_STATE
    receipt['binding'] = digest(receipt)
    atomic_json(folder / 'checkpoint.complete.json', receipt)
    _sync_directory(folder)
    if keep_model_history(plan):_commit_model_receipt(folder,receipt,plan)
    _publish_latest(root,step,plan)
    return receipt


def prune_checkpoints(root,step,plan,*,committed=None):
    root=Path(root)
    folder=root/f'global_step_{step}'
    if committed is None:
        validate_checkpoint(folder,plan)
    elif read_json(folder/'checkpoint.complete.json')!=committed:
        raise ValueError('New checkpoint commit changed before rotation')
    pointer=read_json(root/'latest_complete.json')
    if (pointer.get('step')!=step or pointer.get('plan_binding')!=plan['binding']
            or pointer.get('receipt_sha256')!=file_sha256(folder/'checkpoint.complete.json')):
        raise ValueError('New checkpoint is not the committed restore point')
    # A new durable receipt exists before any old checkpoint can be removed.
    older = []
    for path in root.glob('global_step_*'):
        match = re.fullmatch(r'global_step_([0-9]+)', path.name)
        if (not match or int(match[1]) >= step or path.is_symlink()
                or not (path / 'checkpoint.complete.json').is_file()):continue
        saved = read_json(path / 'checkpoint.complete.json')
        if saved.get('retention_policy') == MINIMAL and saved.get('plan_binding') == plan['binding']:
            older.append(path)
    if keep_model_history(plan):
        # Bind retirement to the already durable newest full restore point.
        replacement={'checkpoint':str(folder),'step':step,'plan_binding':plan['binding'],
            'receipt_sha256':file_sha256(folder/'checkpoint.complete.json')}
        full=_read_full_receipt(folder,plan)
        expected=_model_receipt(folder,full,plan);expected['binding']=digest(expected)
        if read_json(folder/MODEL_RECEIPT)!=expected:raise ValueError('Newest model receipt changed before rotation')
        for path in older:
            # Historical checkpoints without the explicit model-history mark are untouched.
            if read_json(path/'checkpoint.complete.json').get('checkpoint_retention')==ALL_MODELS_LATEST_STATE:
                retire_checkpoint_state(path,plan,replacement=replacement)
    else:
        for path in older:shutil.rmtree(path)


def _sync_directory(folder):
    fd=os.open(folder,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def _publish_latest(root,step,plan):
    atomic_json(root/'latest_complete.json',{'schema':'opd_latest_checkpoint_v1',
        'plan_binding':plan['binding'],'step':step,
        'receipt_sha256':file_sha256(root/f'global_step_{step}'/'checkpoint.complete.json')})
    _sync_directory(root)


def recover_checkpoint(root,plan):
    """An interrupted upstream tracker write must not hide the last valid save."""
    root=Path(root);pointer=root/'latest_complete.json'
    tracker=root/'latest_checkpointed_iteration.txt'
    if pointer.exists():
        saved=read_json(pointer)
        if (saved.get('schema')!='opd_latest_checkpoint_v1'
                or saved.get('plan_binding')!=plan['binding']
                or type(saved.get('step')) is not int or saved['step']<1):
            raise ValueError('Latest checkpoint pointer changed')
        folder=root/f'global_step_{saved["step"]}'
        if file_sha256(folder/'checkpoint.complete.json')!=saved['receipt_sha256']:
            raise ValueError('Latest checkpoint receipt changed')
        validate_checkpoint(folder,plan)
        restore_tracker(root,str(saved['step']).encode())
        return folder
    # First save can crash between its receipt and latest-pointer publication.
    if tracker.exists():
        step=tracker.read_text().strip()
        if not step.isdecimal():raise ValueError('Invalid checkpoint step tracker')
        folder=root/f'global_step_{step}'
        if (folder/'checkpoint.complete.json').exists():
            validate_checkpoint(folder,plan);_publish_latest(root,int(step),plan)
            return folder
        restore_tracker(root,None)
    return None


def restore_tracker(root,previous):
    tracker=Path(root)/'latest_checkpointed_iteration.txt'
    if previous is None:
        tracker.unlink(missing_ok=True)
        return
    temporary=tracker.with_name('.latest-'+uuid.uuid4().hex)
    try:
        with temporary.open('xb') as stream:
            stream.write(previous);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,tracker)
        _sync_directory(tracker.parent)
    finally:temporary.unlink(missing_ok=True)
