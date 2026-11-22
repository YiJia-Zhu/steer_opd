"""SFT weight initialization and export reuse without an OPD resume."""
from copy import deepcopy
from pathlib import Path

import pytest

from steer_opd import sft_opd_baseline
from steer_opd.checkpoint_identity import checkpoint_content_id
from steer_opd.resumable_core import atomic_json, digest
from steer_opd.retention import ALL_MODELS_LATEST_STATE, MINIMAL
from steer_opd.training.retention import SFT_LAYOUT, commit_checkpoint


def test_sft_opd_preview_and_verified_model_only_export(tmp_path, monkeypatch):
    sft = tmp_path / 'sft'
    checkpoint = sft / 'checkpoints/global_step_32'
    checkpoint.mkdir(parents=True)
    (checkpoint.parent / 'latest_checkpointed_iteration.txt').write_text('32')
    source = {'baseline_method': 'sft', 'checkpoint_layout': SFT_LAYOUT, 'output': str(sft),
              'student': 'assets/models/Qwen3-4B', 'settings': {'gpu_pool': [6, 7]},
              'retention_policy': MINIMAL, 'checkpoint_retention': ALL_MODELS_LATEST_STATE}
    source['binding'] = digest(source)
    atomic_json(sft / 'metadata/plan.json', source)
    for rank in range(2):
        for kind in ('model', 'optim', 'extra_state'):
            (checkpoint / f'{kind}_world_size_2_rank_{rank}.pt').write_bytes(kind.encode())
        (checkpoint / f'data_{rank}.pt').write_bytes(b'dataloader')
    commit_checkpoint(checkpoint.parent, 32, source)
    template = {'baseline_method': 'vanilla', 'student': source['student'], 'verl_revision': 'test',
                'tokenizer_paths': {'student': source['student'], 'teacher': 'teacher'},
                'input_sha256': {}, 'implementation_sha256': {}}
    export_root = tmp_path / 'export'
    calls = []

    def merge(folder, output):
        calls.append(folder)
        (output / 'model.safetensors').write_bytes(b'merged weights')

    monkeypatch.setattr(sft_opd_baseline, 'EXPORT_ROOT', export_root)
    monkeypatch.setattr(sft_opd_baseline, 'training_path', Path)
    monkeypatch.setattr(sft_opd_baseline, 'stored_path', str)
    monkeypatch.setattr(sft_opd_baseline, 'merge_sft_model', merge)
    monkeypatch.setattr(sft_opd_baseline, 'checkpoint_content_id',
                        lambda path, cache_dir=None: checkpoint_content_id(path, cache_dir or tmp_path / 'hashes'))

    preview = deepcopy(template)
    assert not sft_opd_baseline.bind_sft_student(preview, checkpoint, materialize=False)
    assert preview['baseline_method'] == 'sft_opd'
    assert preview['student_initialization']['step'] == 32
    assert not export_root.exists()

    prepared = deepcopy(template)
    assert sft_opd_baseline.bind_sft_student(prepared, checkpoint, materialize=True)
    assert prepared['student'] == preview['student'] == prepared['tokenizer_paths']['student']
    assert prepared['student_initialization']['export_content_id']
    assert prepared['tokenizer_paths']['teacher'] == 'teacher'
    assert 'resume_source' not in prepared
    assert sorted(p.name for p in Path(prepared['student']).iterdir()) == ['model.safetensors']
    assert (checkpoint / 'optim_world_size_2_rank_0.pt').read_bytes() == b'optim'
    assert sft_opd_baseline.bind_sft_student(deepcopy(template), checkpoint, materialize=True)
    assert calls == [checkpoint]

    (Path(prepared['student']) / 'model.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError, match='Cached SFT model export changed'):
        sft_opd_baseline.bind_sft_student(deepcopy(template), checkpoint, materialize=True)
