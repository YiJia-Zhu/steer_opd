"""The online reference, teacher-space direction and batch routing of dynamic SV."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest
import torch
from safetensors.torch import save_file

from steer_opd.resumable_core import ROOT
from test_scoring_request_arrays import packet


@pytest.fixture(scope='module')
def dynamic_src():
    return Path(subprocess.check_output(
        [sys.executable, 'scripts/prepare_training_runtime.py', '--dynamic-sv'],
        cwd=ROOT, text=True).strip())


def module(src, name):
    spec = importlib.util.spec_from_file_location(name, src / 'steer_opd/training' / f'{name}.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def fake_module(monkeypatch, name, **values):
    item = ModuleType(name)
    item.__dict__.update(values)
    monkeypatch.setitem(sys.modules, name, item)


@pytest.mark.parametrize('output_mask', [(True, True, True), (True, False, False), (False, True, False)])
def test_dense_linear_backward_matches_autograd(dynamic_src, output_mask):
    linear = module(dynamic_src, 'actor_linear_backward')
    rng = torch.Generator().manual_seed(42)
    inputs = torch.randn(2, 3, 4, generator=rng, dtype=torch.float64).transpose(0, 1).requires_grad_()
    weight = torch.randn(5, 4, generator=rng, dtype=torch.float64, requires_grad=True)
    bias = torch.randn(5, generator=rng, dtype=torch.float64, requires_grad=True)
    grad_output = torch.randn(3, 2, 5, generator=rng, dtype=torch.float64)
    expected = torch.autograd.grad(torch.nn.functional.linear(inputs, weight, bias),
                                  (inputs, weight, bias), grad_output)
    actual = linear.dense_linear_backward(inputs, grad_output, weight, output_mask)
    for enabled, result, reference in zip(output_mask, actual, expected, strict=True):
        if enabled:
            torch.testing.assert_close(result, reference)
        else:
            assert result is None


def test_reference_generation_deduplicates_questions_and_preserves_order(dynamic_src, monkeypatch):
    dynamic = module(dynamic_src, 'dynamic_sv')
    rows = [{'question_id': q, 'question': q, 'prompt_token_ids': [token]}
            for q, token in [('b', 6), ('a', 5)] for _ in range(4)]
    groups = dynamic.group_rollouts(rows, batch_size=2, rollout_n=4)
    fake_module(monkeypatch, 'vllm', SamplingParams=NS)
    fake_module(monkeypatch, 'vllm.sampling_params', RequestOutputKind=NS(FINAL_ONLY='final'))
    sampling = {'n': 1, 'temperature': 1., 'top_p': 1., 'top_k': -1,
                'presence_penalty': 2., 'seed': 42, 'max_tokens': 16}
    calls = []

    def generate(prompts, params, **opts):
        calls.append(prompts)
        assert opts['steering'] is False
        assert all(getattr(params, k) == v for k, v in sampling.items())
        assert params.stop_token_ids == [1] and params.allowed_token_ids == [0, 1, 3]
        return [NS(prompt_token_ids=r['prompt_token_ids'],
                   outputs=[NS(text=f"Teacher {r['prompt_token_ids'][0]}")]) for r in prompts]

    codec = NS(eos={1}, valid_ids=[0, 1, 3], support_size=4)
    assert dynamic.generate_references(NS(generate=generate), codec, groups, sampling,
                                       max_model_len=40) == ['Teacher 6', 'Teacher 5']
    assert calls == [[{'prompt_token_ids': [6]}, {'prompt_token_ids': [5]}]]
    with pytest.raises(ValueError, match='rollouts'):
        dynamic.group_rollouts(rows[:-1], batch_size=2, rollout_n=4)
    rows[1] = dict(rows[1], prompt_token_ids=[5])
    with pytest.raises(ValueError, match='different prompts'):
        dynamic.group_rollouts(rows, batch_size=2, rollout_n=4)


def test_directions_average_teacher_space_differences_per_question(dynamic_src):
    dynamic = module(dynamic_src, 'dynamic_sv')
    vectors = [torch.tensor([float(v), 6.]) for v in (2, 4, 6, 8, 1, 2, 4, 6, 8, 9)]
    # Student means are 5 and 5. Teacher means are 1 and 9, so differences cancel.
    tied = [vectors[:5], None, vectors[5:]]
    kwargs = dict(layer=29, alpha=.1)
    assert dynamic.direction_from_batch(tied, operator='slerp', **kwargs)[0] == {}
    paired = [vectors[:5], None, [*vectors[5:9], torch.tensor([3., 6.])]]
    result, summary = dynamic.direction_from_batch(paired, operator='slerp', **kwargs)
    assert summary == {'usable_questions': 2, 'raw_direction_norm': 3., 'injected': True}
    torch.testing.assert_close(result[29], torch.tensor([1., 0.]))
    result, _ = dynamic.direction_from_batch(paired, operator='additive', **kwargs)
    torch.testing.assert_close(result[29], torch.tensor([.3, 0.]))


def test_static_answer_pooling_reads_both_sets_in_one_teacher_call(dynamic_src, monkeypatch):
    dynamic = module(dynamic_src, 'dynamic_sv')
    group = [{'question': '题', 'source_token_ids': [i]} for i in (1, 2, 3, 4)]
    empty = [{'question': '空', 'source_token_ids': [0]}] * 4

    class TeacherTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == {'tokenize': False, 'add_generation_prompt': True, 'enable_thinking': False}
            return 'P:'

        def __call__(self, text, **kwargs):
            return {'input_ids': [ord(c) for c in text], 'offset_mapping': [(i, i+1) for i in range(len(text))]}

    def batch_decode(rows, **kwargs):
        assert kwargs == {'skip_special_tokens': True, 'clean_up_tokenization_spaces': False}
        return [f' {r[0]}xyz ' if r[0] else ' ' for r in rows]

    student = NS(batch_decode=batch_decode)
    engine = NS(llm_engine=NS(vllm_config=NS(scheduler_config=NS(max_num_batched_tokens=100))))
    calls = []

    def capture(llm, prompts, **opts):
        calls.append(prompts)
        assert llm is engine and 'prompt_logprobs' not in opts and opts['steering'] is False
        expected = ['P:1x', 'P:2x', 'P:3x', 'P:4x', 'P:Ta']
        assert prompts == [{'prompt_token_ids': [ord(c) for c in s]} for s in expected]
        assert all(spec.prompt_positions == [2, 3] for spec in opts['per_prompt_selects'])
        tokens = [p['prompt_token_ids'] for p in prompts]
        return NS(layer_ids=[29], outputs=[NS(prompt_token_ids=p) for p in tokens],
                  sample_positions=lambda _: [2, 3], sample_token_ids=lambda i: tokens[i][2:],
                  sample=lambda i: {29: torch.tensor([[i, 3.], [i + 2., 5.]], dtype=torch.bfloat16)})

    fake_module(monkeypatch, 'easysteer.capture', capture=capture)
    fake_module(monkeypatch, 'vllm.capture', SelectSpec=NS)
    reads = dynamic.capture_answer_means(engine, TeacherTokenizer(), student, [group, empty],
                                         [' Target ', 'yes'], layer=29, hidden_size=2, max_context_len=4)
    assert len(calls) == 1 and reads[1] is None
    torch.testing.assert_close(torch.stack(reads[0]), torch.tensor([[i, 4.] for i in range(1, 6)]))


def test_request_binds_dynamic_sv_to_the_current_step(dynamic_src, tmp_path):
    wire = module(dynamic_src, 'scoring_request')
    request = packet()
    for item in request['rows']:
        item['source']['dynamic_sv_step'] = 1
    first = wire.write_request(tmp_path, request)
    _, rows = wire.read_native_request(tmp_path, first['request_id'], 'plan42', 'teacher')
    assert [(r['question_id'], r['source_token_ids']) for r in rows] == [
        ('q1', item['source']['token_ids']) for item in request['rows']]
    assert all(r['question'] == '中文题目' for r in rows)
    second_request = deepcopy(request)
    for item in second_request['rows']:
        item['source']['dynamic_sv_step'] = 2
    second = wire.write_request(tmp_path, second_request)
    assert first['request_id'] != second['request_id']
    second_request['rows'][0]['source']['dynamic_sv_step'] = 1
    bad = wire.write_request(tmp_path, second_request)
    with pytest.raises(ValueError, match='mixes training steps'):
        wire.read_native_request(tmp_path, bad['request_id'], 'plan42', 'teacher')


@pytest.mark.parametrize('reference_override, expected_limit', [(None, 2048), (4096, 4096)])
def test_dynamic_preview_uses_step_zero_and_online_teacher_references(
        dynamic_src, reference_override, expected_limit, tmp_path):
    config = json.loads((ROOT / 'configs/ablation.json').read_text())
    for role in ('student', 'teacher'):
        model = tmp_path / role
        model.mkdir()
        save_file({'weight': torch.zeros(1)}, model / 'model.safetensors')
        config[role] = str(model)
    config_path = tmp_path / 'ablation.json'
    config_path.write_text(json.dumps(config))
    command = [sys.executable, '-m', 'steer_opd.training.verl_entry', '--check',
               '--config', str(config_path), '--train-output-root', str(tmp_path / 'outputs')]
    if reference_override is not None:
        command += ['--reference-max-tokens', str(reference_override)]
    output = subprocess.check_output(command, cwd=ROOT, text=True,
        env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONPATH': str(dynamic_src)})
    start = output.index('{\n  "plan"')
    plan = json.JSONDecoder().raw_decode(output[start:])[0]['plan']
    assert (plan['sv_mode'], plan['direction']) == ('dynamic', 'S-T')
    assert plan['sv_protocol'] == module(dynamic_src, 'dynamic_sv').PROTOCOL
    assert plan['phase_order'] == ['student_rollout', 'teacher_reference_generation', 'teacher_sv_capture',
                                   'dynamic_sv_teacher_scoring', 'student_update']
    assert plan['sv_reference_sampling'] == {'n': 1, 'temperature': 1., 'top_p': 1., 'top_k': -1,
                                               'presence_penalty': 2., 'seed': 42, 'max_tokens': expected_limit}
    assert f'_Tref{expected_limit}_' in plan['settings']['run_name']
    assert (plan['settings']['max_response_length'], plan['settings']['sv_max_context_len']) == (16384, 2048)
    assert plan['settings']['sv_bank'] is None and plan['assets']['sv_bank_sha256'] is None
    assert (plan['settings']['train_batch_size'], plan['settings']['rollout_n']) == (32, 4)
    assert all(p.startswith(('outputs/step_0_data/', 'assets/datasets/')) for p in plan['input_sha256'])
    assert plan['evaluation_response_format']
