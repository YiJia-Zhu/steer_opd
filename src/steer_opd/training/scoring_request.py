"""Lossless array-backed scoring requests, committed by a small SHA manifest."""
import json
import os
from pathlib import Path
import uuid

import numpy as np

from steer_opd.direction_bank import file_sha256
from steer_opd.resumable_core import atomic_json, digest, read_json

SCHEMA = 'steer_opd_scoring_request_arrays_v1'
ALIGNMENT_FIELDS = {'source_token_ids', 'native_response_ids', 'units', 'alignment_mask',
                    'student_to_unit', 'terminal_unscored_positions', 'response_text'}


def _pack(arrays, name, rows, dtype='int32', tail=(), minimum=0):
    parts = []
    for value in rows:
        raw = np.asarray(value)
        if raw.size and np.dtype(dtype).kind in 'iu':
            limit = np.iinfo(dtype)
            if raw.dtype.kind not in 'iu' or raw.min() < minimum or raw.max() > limit.max:
                raise ValueError(f'Invalid integer scoring array: {name}')
        parts.append(np.asarray(value, dtype=dtype).reshape((-1, *tail)))
    arrays[name] = np.concatenate(parts) if parts else np.empty((0, *tail), dtype=dtype)
    arrays[name+'_offsets'] = np.asarray([0, *np.cumsum([len(p) for p in parts])], dtype=np.int64)


def _encode(request):
    rows = request['rows']
    if not rows:
        raise ValueError('Cannot commit an empty scoring request')
    roles = list(rows[0]['native'])
    alignment_roles = list(rows[0]['alignment']['alignment_roles'])
    reasons = list(dict.fromkeys(u['reason'] for r in rows for u in r['alignment']['units']))
    reason_ids = {reason: index for index, reason in enumerate(reasons)}
    for row in rows:
        if (list(row['native']) != roles
                or row['alignment']['alignment_roles'] != alignment_roles
                or row['source']['token_ids'] != row['alignment']['source_token_ids']
                or any(row['native'][role]['id'] != row['source']['id'] for role in roles)):
            raise ValueError('Scoring source IDs, roles or exact trajectory binding changed')
    arrays = {}
    for field in ('prompt_token_ids', 'token_ids'):
        _pack(arrays, 'source_'+field, [r['source'][field] for r in rows])
    for role in roles:
        for field in ('prompt_token_ids', 'token_ids'):
            _pack(arrays, 'native_'+role+'_'+field, [r['native'][role][field] for r in rows])
    for role in alignment_roles:
        _pack(arrays, 'aligned_'+role+'_ids', [r['alignment']['native_response_ids'][role] for r in rows])
    for field, dtype, minimum in [('alignment_mask', 'bool', 0), ('student_to_unit', 'int32', -1),
                                   ('terminal_unscored_positions', 'int32', 0)]:
        _pack(arrays, field, [r['alignment'][field] for r in rows], dtype, minimum=minimum)
    _pack(arrays, 'response_utf8', [list(r['alignment']['response_text'].encode('utf-8')) for r in rows], 'uint8')
    units = [r['alignment']['units'] for r in rows]
    _pack(arrays, 'unit_bytes', [[[u['byte_start'], u['byte_end']] for u in group] for group in units], tail=(2,))
    _pack(arrays, 'unit_ranges', [[[[*u['ranges'][role]] for role in alignment_roles] for u in group]
                                 for group in units], tail=(len(alignment_roles), 2))
    _pack(arrays, 'unit_valid', [[u['valid'] for u in group] for group in units], 'bool')
    _pack(arrays, 'unit_reason', [[reason_ids[u['reason']] for u in group] for group in units])
    metadata = {'alignment_roles': alignment_roles, 'reasons': reasons, 'rows': [
        {'source': {k: v for k, v in row['source'].items() if k not in ('prompt_token_ids', 'token_ids')},
         'alignment': {k: v for k, v in row['alignment'].items() if k not in ALIGNMENT_FIELDS},
         'native': {role: {k: v for k, v in row['native'][role].items()
                           if k not in ('prompt_token_ids', 'token_ids')} for role in roles}}
        for row in rows]}
    arrays['metadata_utf8'] = np.frombuffer(json.dumps(metadata, ensure_ascii=False,
        separators=(',', ':')).encode('utf-8'), dtype=np.uint8)
    return arrays, roles


def write_request(folder, request):
    """Commit complete alignment provenance and native inputs without JSON tensors.

    The NPZ is uncompressed and contains no pickle/object arrays. A scorer
    validates its SHA but materializes only that role's native input arrays.
    The full alignment remains recoverable for CPU audits and replay checks.
    """
    folder = Path(folder)
    arrays, roles = _encode(request)
    array_folder = folder/'arrays'
    array_folder.mkdir(parents=True, exist_ok=True)
    temporary = array_folder/('.tmp-'+uuid.uuid4().hex+'.npz')
    try:
        with temporary.open('xb') as stream:
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        sha = file_sha256(temporary)
        target = array_folder/(sha+'.npz')
        if target.exists():
            if file_sha256(target) != sha:
                raise ValueError('Existing scoring array content changed')
        else:
            os.replace(temporary, target)
        descriptor = {'path': 'arrays/'+sha+'.npz', 'sha256': sha, 'bytes': target.stat().st_size,
                      'arrays': {name: {'dtype': array.dtype.str, 'shape': list(array.shape)}
                                 for name, array in arrays.items()}}
        manifest = {'schema': SCHEMA, 'plan_binding': request['plan_binding'], 'seed': request['seed'],
                    'source_ids': [row['source']['id'] for row in request['rows']],
                    'alignment_ids': [row['alignment']['alignment_id'] for row in request['rows']],
                    'roles': roles,
                    'native_keys': {role: [row['native'][role]['key'] for row in request['rows']] for role in roles},
                    'payload': descriptor}
        manifest['request_id'] = digest(manifest)
        path = folder/(manifest['request_id']+'.json')
        if path.exists():
            if read_json(path) != manifest:
                raise ValueError('Committed scoring request manifest changed')
        else:
            atomic_json(path, manifest)
        return manifest
    finally:
        temporary.unlink(missing_ok=True)


def _open_request(folder, key, plan_binding):
    if not isinstance(key, str) or len(key) != 64 or any(c not in '0123456789abcdef' for c in key):
        raise ValueError('Invalid scoring request ID')
    folder = Path(folder).resolve()
    manifest = read_json(folder/(key+'.json'))
    if (manifest.get('schema') != SCHEMA or manifest.get('plan_binding') != plan_binding
            or manifest.get('request_id') != key
            or digest({k: v for k, v in manifest.items() if k != 'request_id'}) != key):
        raise ValueError('Scoring request schema, content or plan identity changed')
    descriptor = manifest['payload']
    path = (folder/descriptor['path']).resolve()
    if (descriptor['path'] != 'arrays/'+descriptor['sha256']+'.npz'
            or not path.is_relative_to(folder) or path.stat().st_size != descriptor['bytes']
            or file_sha256(path) != descriptor['sha256']):
        raise ValueError('Scoring request array identity changed')
    count = len(manifest['source_ids'])
    if (not count or len(manifest['alignment_ids']) != count
            or set(manifest['roles']) != set(manifest['native_keys'])
            or any(len(values) != count for values in manifest['native_keys'].values())):
        raise ValueError('Scoring request row identity changed')
    return manifest, path


def _arrays(path, descriptor, names):
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(descriptor['arrays']):
            raise ValueError('Scoring request array fields changed')
        result = {}
        for name in names:
            value = archive[name]
            expected = descriptor['arrays'][name]
            if value.dtype.str != expected['dtype'] or list(value.shape) != expected['shape']:
                raise ValueError('Scoring request array shape or dtype changed')
            result[name] = value
    return result


def _slices(arrays, name, count):
    values, offsets = arrays[name], arrays[name+'_offsets']
    if (offsets.dtype != np.dtype('int64') or offsets.shape != (count+1,)
            or offsets[0] != 0 or offsets[-1] != len(values) or np.any(np.diff(offsets) < 0)):
        raise ValueError('Scoring request ragged offsets changed')
    return [values[a:b] for a, b in zip(offsets[:-1], offsets[1:])]


def read_native_request(folder, key, plan_binding, role):
    """Verify the complete payload; decode only inputs required by one scorer."""
    manifest, path = _open_request(folder, key, plan_binding)
    if role not in manifest['roles']:
        raise ValueError('Scoring role is absent from this request')
    fields = ['native_'+role+'_'+field for field in ('prompt_token_ids', 'token_ids')]
    names = [item for field in fields for item in (field, field+'_offsets')]
    arrays = _arrays(path, manifest['payload'], names)
    count = len(manifest['source_ids'])
    parts = [_slices(arrays, field, count) for field in fields]
    if any(arrays[field].dtype != np.dtype('int32') or arrays[field].ndim != 1
           or np.any(arrays[field] < 0) for field in fields):
        raise ValueError('Invalid native scoring token IDs')
    rows = [{'id': source_id, 'key': manifest['native_keys'][role][i],
             'prompt_token_ids': parts[0][i].tolist(), 'token_ids': parts[1][i].tolist()}
            for i, source_id in enumerate(manifest['source_ids'])]
    return manifest, rows


def read_full_request(folder, key, plan_binding):
    """CPU reconstruction of the original rows, including every alignment field."""
    manifest, path = _open_request(folder, key, plan_binding)
    arrays = _arrays(path, manifest['payload'], manifest['payload']['arrays'])
    metadata = json.loads(arrays['metadata_utf8'].tobytes().decode('utf-8'))
    count = len(manifest['source_ids'])
    if len(metadata['rows']) != count:
        raise ValueError('Scoring metadata row count changed')
    slices = {name: _slices(arrays, name, count) for name in arrays
              if name != 'metadata_utf8' and not name.endswith('_offsets')}
    rows = []
    for i, saved in enumerate(metadata['rows']):
        source = {**saved['source'], **{field: slices['source_'+field][i].tolist()
                   for field in ('prompt_token_ids', 'token_ids')}}
        alignment = {**saved['alignment'], 'source_token_ids': source['token_ids'],
            'native_response_ids': {role: slices['aligned_'+role+'_ids'][i].tolist()
                                    for role in metadata['alignment_roles']},
            'response_text': slices['response_utf8'][i].tobytes().decode('utf-8'),
            **{field: slices[field][i].tolist() for field in
               ('alignment_mask', 'student_to_unit', 'terminal_unscored_positions')}}
        alignment['units'] = [
            {'byte_start': int(bounds[0]), 'byte_end': int(bounds[1]),
             'ranges': {role: ranges[j].tolist() for j, role in enumerate(metadata['alignment_roles'])},
             'valid': bool(valid), 'reason': metadata['reasons'][int(reason)]}
            for bounds, ranges, valid, reason in zip(slices['unit_bytes'][i], slices['unit_ranges'][i],
                                                    slices['unit_valid'][i], slices['unit_reason'][i])]
        if (source['id'] != manifest['source_ids'][i]
                or alignment['alignment_id'] != manifest['alignment_ids'][i]
                or digest({k: v for k, v in alignment.items() if k != 'alignment_id'}) != alignment['alignment_id']):
            raise ValueError('Reconstructed scoring alignment identity changed')
        native = {role: {**saved['native'][role], **{field: slices['native_'+role+'_'+field][i].tolist()
                    for field in ('prompt_token_ids', 'token_ids')}} for role in manifest['roles']}
        rows.append({'source': source, 'alignment': alignment, 'native': native})
    return {'plan_binding': manifest['plan_binding'], 'seed': manifest['seed'], 'rows': rows}
