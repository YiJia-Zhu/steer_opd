"""Retire new pooled SV inputs only after their banks have committed.

Original capture manifests/descriptors stay byte-identical. A separate receipt
binds the intentionally retired pool to fully validated retained bank artifacts.
Missing files without this receipt remain integrity errors.
"""
from pathlib import Path
import os

import torch

from steer_opd.direction_bank import file_sha256
from steer_opd.resumable_core import ROOT, atomic_json, digest, read_json
from steer_opd.retention import MINIMAL

RECEIPT = 'metadata/pool_retirement.json'
SCHEMA = 'sv_pool_retirement_v1'


def usable_ids(folder):
    folder = Path(folder)
    if (folder / RECEIPT).exists():
        return read_json(folder / RECEIPT)['usable_ids']
    pool = torch.load(folder / 'pooled.pt', map_location='cpu', weights_only=True)
    return [qid for qid, record in pool['pooled'].items() if record['usable']]


def verify_retired_capture(store, descriptor, packet):
    from steer_opd.artifact_reuse import verify_files
    from steer_opd.stage_io import project

    folder = project(packet['owner'])
    receipt = read_json(folder / RECEIPT)
    if (not packet['component'].startswith('capture_') or packet.get('external_snapshot')
            or receipt.get('schema') != SCHEMA
            or receipt.get('binding') != digest({k:v for k,v in receipt.items() if k!='binding'})
            or receipt.get('descriptor_sha256') != file_sha256(descriptor)
            or receipt.get('complete_sha256') != packet['complete_sha256']
            or receipt.get('capture_id') != packet['artifact_id']
            or receipt.get('retired_files') != {'pooled.pt': packet['files'].get('pooled.pt')}):
        raise ValueError('SV pool retirement identity changed')
    matched = False
    for bank in receipt['banks']:
        path = project(bank['descriptor'])
        if file_sha256(path) != bank['descriptor_sha256']:
            raise ValueError('Retained SV bank descriptor changed')
        _, saved = store.validate(path)
        if (not saved['component'].startswith('bank_') or saved['artifact_id'] != bank['artifact_id']):
            raise ValueError('Retained SV bank identity changed')
        matched |= saved['contract']['parents'].get('capture') == packet['artifact_id']
    if not matched:
        raise ValueError('No retained bank depends on this retired capture')
    remaining = {name: item for name, item in packet['files'].items()
                 if name != 'pooled.pt' or (folder / name).exists()}
    verify_files(folder, remaining)
    return {name: {'sha256': item['sha256'], 'bytes': item['bytes']}
            for name, item in packet['files'].items()}


def retire_pool(store, capture, banks):
    """Only captures created with an explicit minimal policy can be retired."""
    folder, packet = capture
    marker = read_json(folder / 'complete.json')
    if marker.get('retention_policy') != MINIMAL or packet.get('external_snapshot'):
        return False
    descriptor = ROOT / packet['descriptor_path']
    pool = folder / 'pooled.pt'
    receipt_path = folder / RECEIPT
    if not receipt_path.exists():
        store.validate(descriptor)
        if pool.is_symlink():
            raise ValueError('Refuse to retire a symlinked pool')
        proofs = []
        for _, bank in banks:
            path = ROOT / bank['descriptor_path']
            store.validate(path)
            proofs.append({'descriptor': bank['descriptor_path'],
                           'descriptor_sha256': file_sha256(path), 'artifact_id': bank['artifact_id']})
        receipt={'schema': SCHEMA, 'retention_policy': MINIMAL,
            'capture_id': packet['artifact_id'], 'descriptor_sha256': file_sha256(descriptor),
            'complete_sha256': packet['complete_sha256'],
            'retired_files': {'pooled.pt': packet['files']['pooled.pt']},
            'usable_ids': usable_ids(folder), 'banks': proofs}
        receipt['binding']=digest(receipt)
        atomic_json(receipt_path,receipt)
        fd=os.open(receipt_path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    verify_retired_capture(store, descriptor, packet)
    pool.unlink(missing_ok=True)
    return True
