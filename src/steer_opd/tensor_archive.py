"""Lossless columnar tensor records; Parquet/Zstd and NumPy ZIP archives.

Numeric arrays keep their original dtypes. Per-span dictionaries become typed
columns; metadata retains structure, missing entries, tuple types and bindings.
PyArrow is imported only when a Parquet archive is explicitly used.
"""
import json
from pathlib import Path

import numpy as np
import torch

SCHEMA = 'steer_columnar_tensor_records_v1'
FORMATS = ('parquet_zstd', 'npz_compressed', 'npz')


def _numeric_units(values):
    for item in values:
        if item is None:
            continue
        if not isinstance(item, dict):
            return False
        for key, value in item.items():
            if key == 'top16_matching_indices':
                if not isinstance(value, list) or any(not isinstance(pair, (list, tuple)) or len(pair) != 2 or
                        any(type(x) is not int or not -32768 <= x <= 32767 for x in pair) for pair in value):
                    return False
            elif type(value) not in (int, float) or (type(value) is int and not -2**63 <= value < 2**63):
                return False
    return True


def _pack_value(value, arrays, path):
    def put(suffix, data):
        name = path.replace('%', '%25').replace('.', '%2E')+'/'+suffix+'@'+np.asarray(data).dtype.str
        shape = list(np.asarray(data).shape)
        arrays[name] = np.ascontiguousarray(data)
        return {'name': name, 'shape': shape}
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.layout != torch.strided or tensor.is_quantized:
            raise ValueError('Columnar archives require dense, nonquantized tensors')
        try:
            if tensor.is_complex() or tensor.dtype == torch.float16:
                raise TypeError('Store tensor bits without a Parquet dtype conversion')
            return {'type': 'tensor', **put('tensor', tensor.numpy())}
        except TypeError:
            return {'type': 'tensor', **put('bits', tensor.contiguous().reshape(-1).view(torch.uint8).numpy()),
                    'torch_dtype': str(tensor.dtype).removeprefix('torch.'), 'logical_shape': list(tensor.shape)}
    if isinstance(value, list) and path.endswith(('unit_metrics', 'unit_mean_logps')) and _numeric_units(value):
        keys = list(dict.fromkeys(k for item in value if item for k in item if k != 'top16_matching_indices'))
        present = np.zeros((len(value), len(keys)), dtype=bool)
        integer = np.zeros(present.shape, dtype=bool)
        integer_values = np.zeros(present.shape, dtype=np.int64)
        values = np.zeros(present.shape, dtype=np.float64)
        valid = np.asarray([item is not None for item in value], dtype=bool)
        pairs, offsets, pair_present = [], [0], []
        for i, item in enumerate(value):
            if item is not None:
                for j, key in enumerate(keys):
                    if key in item:
                        present[i, j] = True
                        integer[i, j] = type(item[key]) is int
                        if integer[i, j]:
                            integer_values[i, j] = item[key]
                        else:
                            values[i, j] = item[key]
            current = item.get('top16_matching_indices', []) if item else []
            pairs.extend(current)
            offsets.append(len(pairs))
            pair_present.append(item is not None and 'top16_matching_indices' in item)
        return {'type': 'units', 'keys': keys, 'valid': put('valid', valid),
                'present': put('present', present), 'integer': put('integer', integer),
                'integer_values': put('integer_values', integer_values), 'values': put('values', values),
                'pairs': put('pairs', np.asarray(pairs, dtype=np.int16).reshape(-1, 2)),
                'offsets': put('offsets', np.asarray(offsets, dtype=np.int64)),
                'pair_present': put('pair_present', np.asarray(pair_present, dtype=bool))}
    if isinstance(value, dict):
        return {'type': 'dict', 'items': {k: _pack_value(v, arrays, path+'/'+k) for k, v in value.items()}}
    if isinstance(value, (list, tuple)):
        return {'type': type(value).__name__, 'items': [_pack_value(v, arrays, path+'/'+str(i)) for i, v in enumerate(value)]}
    if value is None or isinstance(value, (bool, int, float, str)):
        return {'type': 'scalar', 'value': value}
    raise TypeError(type(value))


def _unpack_value(spec, arrays):
    def get(item):
        return arrays[item['name']].reshape(item['shape'])
    kind = spec['type']
    if kind == 'scalar':
        return spec['value']
    if kind == 'tensor':
        tensor = torch.from_numpy(get(spec).copy())
        if 'torch_dtype' in spec:
            tensor = tensor.view(getattr(torch, spec['torch_dtype'])).reshape(spec['logical_shape'])
        return tensor
    if kind == 'dict':
        return {k: _unpack_value(v, arrays) for k, v in spec['items'].items()}
    if kind in ('list', 'tuple'):
        values = [_unpack_value(v, arrays) for v in spec['items']]
        return tuple(values) if kind == 'tuple' else values
    valid, present, values = get(spec['valid']), get(spec['present']), get(spec['values'])
    integer = get(spec['integer'])
    integer_values = get(spec['integer_values'])
    pairs, offsets, pair_present = get(spec['pairs']), get(spec['offsets']), get(spec['pair_present'])
    result = []
    for i, ok in enumerate(valid):
        if not ok:
            result.append(None)
            continue
        item = {key: int(integer_values[i, j]) if integer[i, j] else float(values[i, j])
                for j, key in enumerate(spec['keys']) if present[i, j]}
        if pair_present[i]:
            item['top16_matching_indices'] = [tuple(map(int, p)) for p in pairs[offsets[i]:offsets[i+1]]]
        result.append(item)
    return result



def _references(value):
    if isinstance(value, dict):
        if 'name' in value and 'shape' in value:
            yield value['name']
        for item in value.values():
            yield from _references(item)
    elif isinstance(value, list):
        for item in value:
            yield from _references(item)


def _pack_records(records):
    rows, parts = [], []
    for key, record in records.items():
        arrays = {}
        rows.append({'key': key, 'value': _pack_value(record, arrays, 'record')})
        parts.append(arrays)
    columns = sorted(set().union(*(part.keys() for part in parts)))
    packed = {}
    for name in columns:
        dtype = next(part[name].dtype for part in parts if name in part)
        values = [part[name].ravel() if name in part else np.empty(0, dtype=dtype) for part in parts]
        if any(value.dtype != dtype for value in values):
            raise ValueError('Column dtype identity changed')
        packed[name+'.values'] = np.concatenate(values)
        packed[name+'.offsets'] = np.asarray([0, *np.cumsum([len(v) for v in values])], dtype=np.int64)
    metadata = {'schema': SCHEMA, 'rows': rows, 'columns': columns}
    packed['_metadata'] = np.frombuffer(json.dumps(metadata, ensure_ascii=False,
        separators=(',', ':')).encode('utf-8'), dtype=np.uint8)
    return metadata, packed


def _arrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError('Parquet storage requires pyarrow in this project environment') from error
    return pa, pq


def write_archive(path, records, storage):
    """Write a complete shard to an already temporary path; the caller commits it."""
    if storage not in FORMATS or not records:
        raise ValueError('Expected a supported columnar format and nonempty records')
    metadata, arrays = _pack_records(records)
    path = Path(path)
    if storage.startswith('npz'):
        with path.open('wb') as stream:
            (np.savez_compressed if storage == 'npz_compressed' else np.savez)(stream, **arrays)
    else:
        pa, pq = _arrow()
        columns = {'_record_key': pa.array(list(records), type=pa.string()),
                   '_metadata': pa.array([json.dumps(row, ensure_ascii=False,
                        separators=(',', ':')).encode('utf-8') for row in metadata['rows']], type=pa.binary())}
        for name in metadata['columns']:
            columns[name] = pa.LargeListArray.from_arrays(pa.array(arrays[name+'.offsets']),
                                                        pa.array(arrays[name+'.values']))
        table = pa.table(columns).replace_schema_metadata({b'steer_tensor_schema': SCHEMA.encode()})
        pq.write_table(table, path, compression='zstd', compression_level=3, row_group_size=16)


def _choose(metadata, keys, fields):
    if metadata.get('schema') != SCHEMA:
        raise ValueError('Tensor archive schema changed')
    rows = metadata['rows']
    index = {row['key']: i for i, row in enumerate(rows)}
    if len(index) != len(rows):
        raise ValueError('Duplicate record key in tensor archive')
    wanted = list(index) if keys is None else list(keys)
    if any(key not in index for key in wanted) or len(set(wanted)) != len(wanted):
        raise ValueError('Tensor archive record keys changed')
    chosen = []
    for key in wanted:
        value = rows[index[key]]['value']
        if fields is not None:
            if value['type'] != 'dict' or any(field not in value['items'] for field in fields):
                raise ValueError('Requested tensor fields are absent')
            value = {'type': 'dict', 'items': {field: value['items'][field] for field in fields}}
        chosen.append((key, index[key], value))
    required = sorted(set(name for _, _, spec in chosen for name in _references(spec)))
    return chosen, required


def _restore(chosen, arrays):
    result = {}
    for key, i, spec in chosen:
        sliced = {}
        for name in _references(spec):
            offsets, values = arrays[name+'.offsets'], arrays[name+'.values']
            if (offsets.dtype != np.dtype('int64') or offsets.ndim != 1 or len(offsets) <= i+1
                    or offsets[0] != 0 or offsets[-1] != len(values) or np.any(np.diff(offsets) < 0)):
                raise ValueError('Tensor archive offsets changed')
            sliced[name] = values[offsets[i]:offsets[i+1]]
        result[key] = _unpack_value(spec, sliced)
    return result


def read_archive(path, storage, *, keys=None, fields=None):
    """Read selected records/fields without decoding unrelated numeric columns."""
    if storage not in FORMATS:
        raise ValueError('Unknown columnar tensor format')
    if keys is not None and not keys:
        return {}
    if storage.startswith('npz'):
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(archive['_metadata'].tobytes().decode('utf-8'))
            chosen, required = _choose(metadata, keys, fields)
            names = [item for name in required for item in (name+'.values', name+'.offsets')]
            arrays = {name: archive[name] for name in names}
    else:
        _, pq = _arrow()
        if (pq.read_schema(path).metadata or {}).get(b'steer_tensor_schema') != SCHEMA.encode():
            raise ValueError('Tensor archive schema changed')
        filters = None if keys is None else [('_record_key', 'in', list(keys))]
        header = pq.read_table(path, columns=['_metadata'], filters=filters)
        rows = [json.loads(value) for value in header['_metadata'].to_pylist()]
        chosen, required = _choose({'schema': SCHEMA, 'rows': rows}, keys, fields)
        arrays = {}
        if required:
            table = pq.read_table(path, columns=required, filters=filters)
            for name in required:
                column = table[name].combine_chunks()
                offsets = column.offsets.to_numpy(zero_copy_only=False).copy()
                start, end = int(offsets[0]), int(offsets[-1])
                arrays[name+'.values'] = column.values.slice(start, end-start).to_numpy(zero_copy_only=False)
                arrays[name+'.offsets'] = offsets-start
    return _restore(chosen, arrays)

