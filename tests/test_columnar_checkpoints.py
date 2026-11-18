"""Columnar scores keep exact values, partial reads and whole-call atomicity."""
from copy import deepcopy

import pytest
import torch

from steer_opd.experiment_runtime import StageCheckpoint as LegacyCheckpoint
from steer_opd.resumable_core import atomic_json, read_json
from steer_opd.tensor_archive import FORMATS, read_archive, write_archive
from steer_opd.tensor_checkpoint import StageCheckpoint, candidate_checkpoint


def example():
    return {'ids': torch.tensor([[1, 3], [2, 4]], dtype=torch.int32),
        'logps': torch.tensor([[-.1, -.2, -.3], [-.4, -.5, -.6]])[:, ::2],
        'overlap': torch.tensor([.25, float('nan')], dtype=torch.float64),
        'P_overlap': torch.tensor([.123456789012345, float('nan')], dtype=torch.float64),
        'mask': torch.tensor([True, False]), 'scalar': torch.tensor(42),
        'bf16': torch.tensor([1.5, float('nan')], dtype=torch.bfloat16),
        'fp16': torch.tensor([.25, float('nan')], dtype=torch.float16),
        'complex': torch.tensor([1+2j], dtype=torch.complex64),
        'broadcast_mean_logps': {'student': torch.tensor([-1., -2.])},
        'unit_metrics': [None, {'overlap': .5, 'student_shared_mass': 0,
                              'count': 2**60+1, 'top16_matching_indices': [(1, 2), (2, 0)]}],
        'unit_mean_logps': [{}, {'student': -.123456789012345, 'teacher': -2.}],
        'metadata': {'unicode': '中文🙂', 'tuple': (None, True, 2**100), 'empty': []}}


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert a.dtype == b.dtype and a.shape == b.shape
        assert a.contiguous().reshape(-1).view(torch.uint8).equal(b.contiguous().reshape(-1).view(torch.uint8))
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert type(a) == type(b) and len(a) == len(b)
        for left, right in zip(a, b):
            equal(left, right)
    else:
        assert type(a) == type(b) and a == b


@pytest.mark.parametrize('storage', FORMATS)
def test_archive_exact_roundtrip_mixed_dtypes_and_subset_fields(tmp_path, storage):
    values = {'q1': example(), 'q2': example()}
    values['q2']['ids'] = values['q2']['ids'].long()
    values['q2']['field.with.dot'] = torch.tensor([.125], dtype=torch.float64)
    path = tmp_path/'archive'
    write_archive(path, values, storage)
    equal(values, read_archive(path, storage))
    selected = read_archive(path, storage, keys=['q2'], fields=['overlap', 'P_overlap'])
    equal(selected, {'q2': {k: values['q2'][k] for k in ('overlap', 'P_overlap')}})
    with pytest.raises(ValueError, match='keys changed'):
        read_archive(path, storage, keys=['unknown'])


@pytest.mark.parametrize('storage', FORMATS)
def test_checkpoint_shards_manifest_auto_resume_and_column_reads(tmp_path, storage):
    records = {f'q{i}': example() for i in range(3)}
    store = StageCheckpoint(tmp_path, 'binding', tensor_format=storage, shard_size=2)
    extra = tmp_path/'summary.json'
    atomic_json(extra, {'count': 3})
    store.save_tensors(records, [extra])
    manifest = read_json(tmp_path/'complete.json')
    assert len(manifest['record_shards']) == 2
    assert len(manifest['files']) == 3
    identity = (tmp_path/'stage_input.json').read_bytes()
    loaded = StageCheckpoint(tmp_path, 'binding')
    equal(records, loaded.load_tensors(list(records)))
    equal(loaded.load_tensor_fields(['q2'], ['overlap']), {'q2': {'overlap': records['q2']['overlap']}})
    assert (tmp_path/'stage_input.json').read_bytes() == identity
    with pytest.raises(ValueError, match='response keys changed'):
        loaded.load_tensors(['q2'])
    with pytest.raises(ValueError, match='cannot be overwritten'):
        loaded.save_tensors(records)
    with pytest.raises(ValueError, match='identity changed'):
        StageCheckpoint(tmp_path, 'binding', tensor_format=storage, shard_size=3)


@pytest.mark.parametrize('compression', ['none', 'gzip'])
def test_existing_legacy_output_reads_without_rewriting_any_bytes(tmp_path, compression):
    records = {'q': {'value': torch.tensor([.25, float('nan')])}}
    LegacyCheckpoint(tmp_path, 'same', tensor_compression=compression).save_tensors(records)
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.iterdir()}
    equal(records, StageCheckpoint(tmp_path, 'same').load_tensors(['q']))
    equal(records, candidate_checkpoint(tmp_path, 'same', {}).load_tensors(['q']))
    with pytest.raises(ValueError, match='identity changed'):
        StageCheckpoint(tmp_path, 'same', tensor_format='parquet_zstd')
    after = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.iterdir()}
    assert before == after


@pytest.mark.parametrize('storage', FORMATS)
def test_corrupted_shard_rejected_even_for_selected_column(tmp_path, storage):
    StageCheckpoint(tmp_path, 'fixed', tensor_format=storage).save_tensors({'q': example()})
    manifest = read_json(tmp_path/'complete.json')
    path = tmp_path/manifest['record_shards'][0]['file']
    with path.open('r+b') as stream:
        stream.seek(path.stat().st_size//2)
        stream.write(b'changed')
    with pytest.raises(ValueError, match='missing or changed'):
        StageCheckpoint(tmp_path, 'fixed').load_tensor_fields(['q'], ['overlap'])


def test_interrupted_shard_write_cannot_be_loaded_until_whole_call_commits(tmp_path, monkeypatch):
    import steer_opd.tensor_checkpoint as module
    original = module.write_archive
    calls = []
    def interrupted(*args):
        calls.append(True)
        if len(calls) == 2:
            raise KeyboardInterrupt()
        return original(*args)
    monkeypatch.setattr(module, 'write_archive', interrupted)
    records = {'q1': example(), 'q2': example()}
    stage = StageCheckpoint(tmp_path, 'fixed', tensor_format='npz_compressed', shard_size=1)
    with pytest.raises(KeyboardInterrupt):
        stage.save_tensors(records)
    assert not (tmp_path/'complete.json').exists() and stage.load_tensors(list(records)) is None
    monkeypatch.setattr(module, 'write_archive', original)
    stage.save_tensors(records)
    equal(records, stage.load_tensors(list(records)))


def test_parquet_selected_field_does_not_decode_other_numeric_columns(tmp_path, monkeypatch):
    import pyarrow.parquet as pq
    original = pq.read_table
    stage = StageCheckpoint(tmp_path, 'fixed', tensor_format='parquet_zstd')
    stage.save_tensors({'q': example()})
    columns = []
    def inspect(*args, **kwargs):
        columns.extend(kwargs.get('columns') or [])
        return original(*args, **kwargs)
    monkeypatch.setattr(pq, 'read_table', inspect)
    stage.load_tensor_fields(['q'], ['overlap'])
    assert columns and all(name == '_metadata' or name.startswith('record/overlap/') for name in columns)


def test_launcher_defaults_to_parquet_and_accepts_legacy_options(monkeypatch):
    import importlib.util
    from steer_opd.resumable_core import ROOT
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    spec = importlib.util.spec_from_file_location('columnar_launcher', ROOT/'scripts/launch_ablation.py')
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    assert launcher.parser().parse_args([]).candidate_tensor_compression == 'parquet_zstd'
    for value in ('npz_compressed', 'npz', 'parquet_zstd', 'pt', 'pt_gzip'):
        cfg = launcher.build_config(launcher.parser().parse_args(['--candidate-tensor-format', value]))
        assert cfg['candidate_tensor_compression'] == value
    assert launcher.parser().parse_args(['--candidate-tensor-compression', 'gzip']).candidate_tensor_compression == 'gzip'
