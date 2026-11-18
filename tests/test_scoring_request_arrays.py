"""Lossless scorer wire format, identity checks and client cache integration."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from steer_opd.direction_bank import file_sha256
from steer_opd.resumable_core import atomic_json, atomic_torch, digest, read_json
from steer_opd.training.config import TrainingLossConfig
from steer_opd.training.losses import training_alignment
from steer_opd.training.scoring_request import SCHEMA, read_full_request, read_native_request, write_request
from test_compass_alignment import ToyCodec


def packet(kind='traditional_opd'):
    student = ToyCodec({0: b'<eos>', 1: b'a', 2: b'b', 3: b'c', 4: b'\xe4', 5: b'\xb8', 6: b'\xad'})
    teacher = ToyCodec({0: b'<eos>', 11: b'ab', 12: b'c', 13: '中'.encode()})
    reference = ToyCodec({0: b'<eos>', 21: b'a', 22: b'bc', 23: '中'.encode()})
    settings = TrainingLossConfig(loss_type=kind)
    rows = []
    for i, tokens in enumerate(([1, 2, 3, 4, 5, 6, 0], [0])):
        # Duplicate question/trajectory IDs are legal for a question's four rollouts.
        source = {'id': 'same-question', 'question_id': 'q1', 'question': '中文题目',
                  'prompt_token_ids': [7, 8], 'token_ids': tokens}
        alignment = training_alignment(tokens, student, teacher, settings=settings,
                    reference_codec=reference if kind == 'compass_opd' else None)
        if alignment['units']:
            # Exercise preservation of excluded control spans, not only EOS.
            unit = alignment['units'][0]
            unit['valid'] = False
            unit['reason'] = 'control_token_in_span'
            a, b = unit['ranges']['student']
            alignment['alignment_mask'][a:b] = [False]*(b-a)
            alignment['student_to_unit'][a:b] = [-1]*(b-a)
            alignment['alignment_id'] = digest({k: v for k, v in alignment.items() if k != 'alignment_id'})
        native = {role: {'id': source['id'], 'key': source['id'], 'prompt_token_ids': [40+i, 41],
                         'token_ids': alignment['native_response_ids'][role]}
                  for role in alignment['alignment_roles'] if role != 'student'}
        if kind == 'compass_opd':
            native['student_anchor'] = {'id': source['id'], 'key': source['id'],
                'prompt_token_ids': source['prompt_token_ids'], 'token_ids': source['token_ids']}
        rows.append({'source': source, 'alignment': alignment, 'native': native})
    return {'plan_binding': 'plan42', 'seed': 42, 'rows': rows}


@pytest.mark.parametrize('kind', ['traditional_opd', 'compass_opd'])
def test_full_roundtrip_preserves_all_ids_utf8_masks_alignment_and_native_inputs(tmp_path, kind):
    request = packet(kind)
    before = deepcopy(request)
    manifest = write_request(tmp_path, request)
    assert request == before
    assert read_full_request(tmp_path, manifest['request_id'], 'plan42') == request
    assert manifest['schema'] == SCHEMA and 'rows' not in manifest
    for role in manifest['roles']:
        loaded, rows = read_native_request(tmp_path, manifest['request_id'], 'plan42', role)
        assert loaded == manifest
        assert rows == [row['native'][role] for row in request['rows']]
    with np.load(tmp_path/manifest['payload']['path'], allow_pickle=False) as archive:
        assert all(archive[name].dtype.kind != 'O' for name in archive.files)
        assert archive['native_teacher_token_ids'].dtype == np.dtype('int32')
        assert archive['alignment_mask'].dtype == np.dtype('bool')
    # Same contents produce the same request and preserve committed files.
    paths = [tmp_path/(manifest['request_id']+'.json'), tmp_path/manifest['payload']['path']]
    stats = [path.stat().st_mtime_ns for path in paths]
    assert write_request(tmp_path, request) == manifest
    assert stats == [path.stat().st_mtime_ns for path in paths]


def test_native_reader_does_not_decode_full_alignment_or_metadata(tmp_path, monkeypatch):
    manifest = write_request(tmp_path, packet())
    import steer_opd.training.scoring_request as wire
    original = wire._arrays
    def inspect(path, descriptor, names):
        assert all(name.startswith('native_teacher_') for name in names)
        return original(path, descriptor, names)
    monkeypatch.setattr(wire, '_arrays', inspect)
    read_native_request(tmp_path, manifest['request_id'], 'plan42', 'teacher')


@pytest.mark.parametrize('change', ['plan', 'manifest', 'payload', 'missing', 'role'])
def test_wrong_identity_or_corrupted_inputs_are_rejected_before_scoring(tmp_path, change):
    manifest = write_request(tmp_path, packet())
    key = manifest['request_id']
    expected_plan, role = 'plan42', 'teacher'
    if change == 'plan':
        expected_plan = 'another-checkpoint-plan'
    elif change == 'manifest':
        manifest['source_ids'].reverse()
        manifest['seed'] = 43
        atomic_json(tmp_path/(key+'.json'), manifest)
    elif change == 'payload':
        path = tmp_path/manifest['payload']['path']
        with path.open('r+b') as stream:
            stream.seek(50)
            stream.write(b'changed')
    elif change == 'missing':
        (tmp_path/manifest['payload']['path']).unlink()
    else:
        role = 'reference'
    with pytest.raises((ValueError, FileNotFoundError)):
        read_native_request(tmp_path, key, expected_plan, role)


def test_native_token_change_changes_identity_and_int32_overflow_is_rejected(tmp_path):
    request = packet()
    first = write_request(tmp_path, request)
    request['rows'][0]['native']['teacher']['prompt_token_ids'][0] += 1
    second = write_request(tmp_path, request)
    assert first['request_id'] != second['request_id']
    request['rows'][0]['native']['teacher']['token_ids'] = [2**32]
    with pytest.raises(ValueError, match='integer scoring array'):
        write_request(tmp_path, request)


def test_legacy_json_request_is_not_mistaken_for_array_protocol(tmp_path):
    request = packet()
    key = digest(request)
    atomic_json(tmp_path/(key+'.json'), {**request, 'request_id': key})
    with pytest.raises(ValueError, match='schema, content or plan'):
        read_native_request(tmp_path, key, 'plan42', 'teacher')


@pytest.mark.parametrize('kind', ['traditional_opd', 'compass_opd'])
@pytest.mark.parametrize('policy',[None,'minimal_v1'])
def test_client_scorer_roundtrip_and_cache_keep_original_alignment(tmp_path, monkeypatch, kind,policy):
    import steer_opd.training.scorer_client as client_module
    from steer_opd.training.overlap_monitor import save_top16
    request = packet(kind)
    monkeypatch.setattr(client_module, 'prepare_request', lambda *args: request)
    client = object.__new__(client_module.ScorerClient)
    client.folder = tmp_path/'scorer'
    client.plan = {'binding': 'plan42', 'output': str(tmp_path), 'assets': {'model': 'frozen'},
                   'settings': {'monitor_overlap': True}}
    if policy:client.plan['retention_policy']=policy
    client.pending_request=None
    from steer_opd.training.retention import batch_root,reset_batch_workspace
    reset_batch_workspace(client.folder,client.plan)
    client.codecs = {}
    client.roles = list(request['rows'][0]['native'])
    client.states = dict.fromkeys(client.roles, 'asleep')
    client.shared_phase = True
    calls = []
    client._wait = lambda predicate: predicate() or pytest.fail('Missing committed scorer reply')
    def control(role, action, key=None):
        calls.append((role, action))
        client.states[role] = 'awake' if action == 'wake' else 'asleep'
        if action != 'wake':
            return
        manifest, native = read_native_request(batch_root(client.folder,client.plan)/'requests', key, 'plan42', role)
        assert native == [row['native'][role] for row in request['rows']]
        target = batch_root(client.folder,client.plan)/'cache'/role/key
        values = [torch.full((len(row['token_ids']),), -.5) for row in native]
        atomic_torch(target/'observed_logps.pt', values)
        receipt = {'request_schema': SCHEMA, 'request_id': key, 'plan_binding': 'plan42',
                   'role': role, 'source_ids': manifest['source_ids'], 'assets': client.plan['assets']}
        if role == 'teacher':
            top16 = [{'ids': torch.zeros((len(v), 16), dtype=torch.int32),
                      'logps': torch.full((len(v), 16), -3.), 'sampled_logps': v} for v in values]
            save_top16(target/'top16.pt.gz', top16)
            atomic_json(target/'top16.complete.json', {**receipt, 'sha256': file_sha256(target/'top16.pt.gz')})
        atomic_json(target/'complete.json', {**receipt, 'sha256': file_sha256(target/'observed_logps.pt')})
    client.control = control
    rows = [row['source'] for row in request['rows']]
    first = client.score(rows)
    call_count = len(calls)
    second = client.score(rows)
    assert len(calls) == call_count == 2*len(client.roles)
    for reply, cached, original in zip(first, second, request['rows']):
        assert reply['alignment'] == cached['alignment'] == original['alignment']
        torch.testing.assert_close(reply['teacher_logps'], cached['teacher_logps'])
        assert 'teacher_top16' in reply
        if kind == 'compass_opd':
            assert 'reference_logps' in reply and 'student_anchor_logps' in reply
    if policy:
        client.shared_phase=False
        client.release_batch(1)
        assert not (batch_root(client.folder,client.plan)/'cache').exists()
        assert not (batch_root(client.folder,client.plan)/'requests').exists()
        # RAM replies remain usable after their temporary files are gone.
        assert first[0]['teacher_logps'].numel()>0


def test_training_plan_binds_array_schema_and_rejects_old_protocol():
    from steer_opd.resumable_core import ROOT
    from steer_opd.training.verl_plan import build_verl_plan, verl_overrides
    plan = build_verl_plan(read_json(ROOT/'configs/ablation.json'))
    assert plan['scoring_request_schema'] == SCHEMA
    assert 'src/steer_opd/training/scoring_request.py' in plan['implementation_sha256']
    del plan['scoring_request_schema']
    with pytest.raises(ValueError, match='new training plan'):
        verl_overrides(plan)
