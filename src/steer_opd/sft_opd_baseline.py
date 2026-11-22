"""Bind an OPD initialization to a saved SFT model, without its optimizer state."""
import fcntl
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from steer_opd.checkpoint_identity import checkpoint_content_id
from steer_opd.direction_bank import file_sha256
from steer_opd.resumable_core import ROOT, atomic_json, digest, read_json
from steer_opd.training.paths import stored_path, training_path
from steer_opd.training.retention import MODEL_RECEIPT, SFT_LAYOUT, validate_model_checkpoint
from steer_opd.training.storage import require_training_space


EXPORT_ROOT = ROOT / '.runtime/sft_opd_models'


def merge_sft_model(checkpoint, target):
    from steer_opd.training.verl_entry import activate_verl

    activate_verl()
    subprocess.run([sys.executable, '-m', 'verl.model_merger', 'merge', '--backend', 'fsdp',
                    '--local_dir', str(checkpoint), '--target_dir', str(target), '--use_cpu_initialization'],
                   env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}, cwd=ROOT, check=True)


def model_export(checkpoint, target, identity, model_bytes, *, materialize):
    model = target / 'model'
    if materialize and not target.exists():
        EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
        with target.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not target.exists():
                require_training_space(EXPORT_ROOT, model_bytes)
                with TemporaryDirectory(dir=EXPORT_ROOT, prefix='.export-') as temporary:
                    staged = Path(temporary) / 'ready'
                    (staged / 'model').mkdir(parents=True)
                    merge_sft_model(checkpoint, staged / 'model')
                    content_id = checkpoint_content_id(staged / 'model', cache_dir=Path(temporary) / 'hashes')
                    atomic_json(staged / 'complete.json', {'source': identity, 'model_content_id': content_id})
                    staged.rename(target)
    if not target.exists():
        return None
    record = read_json(target / 'complete.json')
    if record['source'] != identity or record['model_content_id'] != checkpoint_content_id(model):
        raise ValueError('Cached SFT model export changed')
    return record['model_content_id']


def bind_sft_student(plan, checkpoint, *, materialize):
    """Choose the SFT model as the initial actor and rollout model for vanilla OPD."""
    if plan['baseline_method'] != 'vanilla':
        raise ValueError('This SFT initialization is for standard OPD')
    folder = training_path(checkpoint)
    source_path = folder.parent.parent / 'metadata/plan.json'
    source = read_json(source_path)
    if (source.get('baseline_method') != 'sft' or source.get('checkpoint_layout') != SFT_LAYOUT
            or source.get('binding') != digest({key: val for key, val in source.items() if key != 'binding'})
            or folder.parent != training_path(source['output']) / 'checkpoints'):
        raise ValueError('Expected a checkpoint owned by a frozen SFT run')
    if (ROOT / source['student']).resolve() != (ROOT / plan['student']).resolve():
        raise ValueError('SFT checkpoint and OPD plan use different student models')
    receipt = validate_model_checkpoint(folder, source)
    if folder.name != f"global_step_{receipt['step']}":
        raise ValueError('SFT checkpoint directory does not match its committed step')

    receipt_path = folder / MODEL_RECEIPT
    identity = {
        'mode': 'sft_model_only',
        'checkpoint': stored_path(folder),
        'step': receipt['step'],
        'source_plan_binding': source['binding'],
        'model_receipt_sha256': file_sha256(receipt_path),
        'export_format': 'verl_fsdp_to_hf_bfloat16',
        'export_verl_revision': plan['verl_revision'],
    }
    target = EXPORT_ROOT / digest(identity)
    content_id = model_export(folder, target, identity, sum(item['bytes'] for item in receipt['files'].values()),
                              materialize=materialize)
    model = stored_path(target / 'model')
    plan['baseline_method'] = 'sft_opd'
    plan['student'] = model
    plan['tokenizer_paths'] = {**plan['tokenizer_paths'], 'student': model}
    plan['student_initialization'] = {**identity, 'export_content_id': content_id}
    plan['input_sha256'].update({stored_path(source_path): file_sha256(source_path),
                                  stored_path(receipt_path): identity['model_receipt_sha256']})
    implementation = Path(__file__).resolve()
    plan['implementation_sha256'][str(implementation.relative_to(ROOT))] = file_sha256(implementation)
    return content_id is not None
