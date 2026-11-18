"""Read-only input benchmark; lossless sample files live in a temporary folder."""
from pathlib import Path
from tempfile import TemporaryDirectory
import gzip
import io
import json
import math
import statistics
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from steer_opd.resumable_core import ROOT, atomic_json


def pack(value, arrays, path):
    def put(suffix, data):
        name = path+'/'+suffix
        shape = list(np.asarray(data).shape)
        arrays[name] = np.ascontiguousarray(data)
        return {'name': name, 'shape': shape}
    if isinstance(value, torch.Tensor):
        return {'type': 'tensor', **put('tensor', value.detach().cpu().numpy())}
    if isinstance(value, list) and path.endswith(('unit_metrics', 'unit_mean_logps')):
        keys = list(dict.fromkeys(k for item in value if item for k in item if k != 'top16_matching_indices'))
        present = np.zeros((len(value), len(keys)), dtype=bool)
        integer = np.zeros(present.shape, dtype=bool)
        values = np.zeros(present.shape, dtype=np.float64)
        valid = np.asarray([item is not None for item in value], dtype=bool)
        pairs, offsets, pair_present = [], [0], []
        for i, item in enumerate(value):
            if item is not None:
                for j, key in enumerate(keys):
                    if key in item:
                        present[i, j] = True
                        integer[i, j] = type(item[key]) is int
                        values[i, j] = item[key]
            current = item.get('top16_matching_indices', []) if item else []
            pairs.extend(current)
            offsets.append(len(pairs))
            pair_present.append(item is not None and 'top16_matching_indices' in item)
        return {'type': 'units', 'keys': keys, 'valid': put('valid', valid),
                'present': put('present', present), 'integer': put('integer', integer), 'values': put('values', values),
                'pairs': put('pairs', np.asarray(pairs, dtype=np.int16).reshape(-1, 2)),
                'offsets': put('offsets', np.asarray(offsets, dtype=np.int64)),
                'pair_present': put('pair_present', np.asarray(pair_present, dtype=bool))}
    if isinstance(value, dict):
        return {'type': 'dict', 'items': {k: pack(v, arrays, path+'/'+k) for k, v in value.items()}}
    if isinstance(value, (list, tuple)):
        return {'type': type(value).__name__, 'items': [pack(v, arrays, path+'/'+str(i)) for i, v in enumerate(value)]}
    if value is None or isinstance(value, (bool, int, float, str)):
        return {'type': 'scalar', 'value': value}
    raise TypeError(type(value))


def unpack(spec, arrays):
    def get(item):
        return arrays[item['name']].reshape(item['shape'])
    kind = spec['type']
    if kind == 'scalar':
        return spec['value']
    if kind == 'tensor':
        return torch.from_numpy(get(spec).copy())
    if kind == 'dict':
        return {k: unpack(v, arrays) for k, v in spec['items'].items()}
    if kind in ('list', 'tuple'):
        values = [unpack(v, arrays) for v in spec['items']]
        return tuple(values) if kind == 'tuple' else values
    valid, present, values = get(spec['valid']), get(spec['present']), get(spec['values'])
    integer = get(spec['integer'])
    pairs, offsets, pair_present = get(spec['pairs']), get(spec['offsets']), get(spec['pair_present'])
    result = []
    for i, ok in enumerate(valid):
        if not ok:
            result.append(None)
            continue
        item = {key: int(values[i, j]) if integer[i, j] else float(values[i, j])
                for j, key in enumerate(spec['keys']) if present[i, j]}
        if pair_present[i]:
            item['top16_matching_indices'] = [tuple(map(int, p)) for p in pairs[offsets[i]:offsets[i+1]]]
        result.append(item)
    return result


def assert_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=True)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert type(a) == type(b) and len(a) == len(b)
        for left, right in zip(a, b):
            assert_equal(left, right)
    elif isinstance(a, float) and math.isnan(a):
        assert math.isnan(b)
    else:
        assert type(a) == type(b) and a == b


def measure(call, rounds=3):
    call()
    durations = []
    for _ in range(rounds):
        start = time.perf_counter()
        value = call()
        durations.append(time.perf_counter()-start)
        del value
    return {'median_seconds': statistics.median(durations), 'seconds': durations}


def main():
    pa.set_cpu_count(4)
    run = ROOT/'outputs/step_3_overlap/param/T_Qwen3.5-35B-A3B_S_Qwen3-4B/OpenThoughts3-1.2M/shared/n1000_additive0.5_S26_T39_seed42_st_all_global_v2'
    pool = sorted([p for split in ('select', 'confirm') for p in (run/split/'teacher/baseline').glob('*.pt.gz')],
                  key=lambda p: p.stat().st_size)
    selected = [pool[int((i+.5)*len(pool)/16)] for i in range(16)]
    paths = [p.parent.parent/candidate/p.name for p in selected
             for candidate in ('baseline', 'teacher_unsteered_replay', 'scheme_05')]
    def old_load():
        rows = []
        for path in paths:
            # Match StageCheckpoint.load_tensors, avoiding GzipFile's expensive
            # random seeks through the nested torch ZIP archive.
            stream = io.BytesIO(gzip.decompress(path.read_bytes()))
            rows.append(torch.load(stream, map_location='cpu', weights_only=True))
        return rows
    records = old_load()
    print('Loaded '+str(len(records))+' real records; packing all fields losslessly', flush=True)
    started = time.perf_counter()
    heads, parts = [], []
    for record in records:
        arrays = {}
        heads.append(pack(record, arrays, 'record'))
        parts.append(arrays)
    columns = sorted(set().union(*(item.keys() for item in parts)))
    packed = {'_metadata': np.frombuffer(json.dumps(heads, ensure_ascii=False, separators=(',', ':')).encode(), dtype=np.uint8)}
    for name in columns:
        dtype = next(part[name].dtype for part in parts if name in part)
        values = [part[name].ravel() if name in part else np.empty(0, dtype=dtype) for part in parts]
        assert all(v.dtype == dtype for v in values)
        packed[name+'.values'] = np.concatenate(values)
        packed[name+'.offsets'] = np.asarray([0, *np.cumsum([len(v) for v in values])], dtype=np.int64)
    encode_seconds = time.perf_counter()-started

    def restore(arrays):
        head = json.loads(arrays['_metadata'].tobytes())
        rows = []
        for i, meta in enumerate(head):
            sliced = {name: arrays[name+'.values'][arrays[name+'.offsets'][i]:arrays[name+'.offsets'][i+1]] for name in columns}
            rows.append(unpack(meta, sliced))
        return rows
    metric_columns = ['record/overlap/tensor', 'record/P_overlap/tensor']
    def metric_reduce(arrays):
        a, b = [arrays[name+'.values'] for name in metric_columns]
        valid = np.isfinite(a) & np.isfinite(b)
        return [int(valid.sum()), float(a[valid].sum()), float(b[valid].sum())]
    def old_metrics():
        rows = old_load()
        a = np.concatenate([row['overlap'].numpy() for row in rows])
        b = np.concatenate([row['P_overlap'].numpy() for row in rows])
        valid = np.isfinite(a) & np.isfinite(b)
        return [int(valid.sum()), float(a[valid].sum()), float(b[valid].sum())]

    results = {'pt_gzip': {'bytes': sum(p.stat().st_size for p in paths),
        'full_object_read': measure(old_load), 'metrics_read': measure(old_metrics)}}
    with TemporaryDirectory(prefix='step3_formats_', dir=ROOT/'.runtime/tmp') as temporary:
        folder = Path(temporary)
        for compressed in (False, True):
            name = 'npz_deflate' if compressed else 'npz_uncompressed'
            path = folder/(name+'.npz')
            start = time.perf_counter()
            (np.savez_compressed if compressed else np.savez)(path, **packed)
            write_seconds = time.perf_counter()-start
            def load(selected=None):
                with np.load(path, allow_pickle=False) as archive:
                    return {key: archive[key] for key in (selected or archive.files)}
            rebuilt = restore(load())
            for old, new in zip(records, rebuilt):
                assert_equal(old, new)
            del rebuilt
            wanted = [name+'.values' for name in metric_columns]
            np.testing.assert_equal(metric_reduce(load(wanted)), old_metrics())
            results[name] = {'bytes': path.stat().st_size, 'write_seconds_excluding_encode': write_seconds,
                'array_read': measure(load), 'full_object_read': measure(lambda: restore(load())),
                'metrics_read': measure(lambda: metric_reduce(load(wanted))), 'lossless_roundtrip': True}
            print(name+' verified', flush=True)

        arrow_columns = {'_metadata': pa.array([json.dumps(h, ensure_ascii=False, separators=(',', ':')).encode() for h in heads], type=pa.binary())}
        for name in columns:
            arrow_columns[name] = pa.LargeListArray.from_arrays(pa.array(packed[name+'.offsets']), pa.array(packed[name+'.values']))
        table = pa.table(arrow_columns)
        path = folder/'parquet_zstd.parquet'
        start = time.perf_counter()
        pq.write_table(table, path, compression='zstd', compression_level=3, row_group_size=16)
        write_seconds = time.perf_counter()-start
        def parquet_load(selected=None):
            table = pq.read_table(path, columns=selected)
            arrays = {}
            if '_metadata' in table.column_names:
                arrays['_metadata'] = np.frombuffer(('['+','.join(v.decode() for v in table['_metadata'].to_pylist())+']').encode(), dtype=np.uint8)
            for name in table.column_names:
                if name == '_metadata':
                    continue
                field = table[name].combine_chunks()
                arrays[name+'.values'] = field.values.to_numpy(zero_copy_only=False)
                arrays[name+'.offsets'] = field.offsets.to_numpy(zero_copy_only=False)
            return arrays
        rebuilt = restore(parquet_load())
        for old, new in zip(records, rebuilt):
            assert_equal(old, new)
        del rebuilt
        np.testing.assert_equal(metric_reduce(parquet_load(metric_columns)), old_metrics())
        results['parquet_zstd3'] = {'bytes': path.stat().st_size, 'write_seconds_excluding_encode': write_seconds,
            'array_read': measure(parquet_load), 'full_object_read': measure(lambda: restore(parquet_load())),
            'metrics_read': measure(lambda: metric_reduce(parquet_load(metric_columns))), 'lossless_roundtrip': True}
        print('parquet_zstd3 verified', flush=True)

    report = {'scope': 'CPU sample benchmark; originals untouched; hot page cache, 1 warmup + 3 timed reads; all fields and original float precision preserved',
              'sample': {'parameter': str(run.relative_to(ROOT)), 'records': len(records), 'questions': len(selected),
                         'selection': '16 baseline compressed-size quantiles, each baseline/replay/SV; select+confirm pool',
                         'teacher_tokens': sum(len(r['sampled_logps']) for r in records),
                         'source_files': [str(p.relative_to(ROOT)) for p in paths]},
              'versions': {'numpy': np.__version__, 'pyarrow': pa.__version__, 'torch': torch.__version__},
              'common_array_encoding_seconds': encode_seconds, 'results': results,
              'temporary_format_files_retained': False, 'production_reader_changed': False, 'gpu_started': False}
    atomic_json(ROOT/'audit/step3_format_comparison_20261118.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'sample'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
