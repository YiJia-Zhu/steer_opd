"""Cross-tokenizer span semantics, metric identity and four-way integration."""

import copy
import hashlib
import importlib
import json
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

from steer_opd.compass_alignment import ByteLevelCodec, METRIC, _byte_decoder, align_response, aligned_metrics, span_scores
from steer_opd.compass_banks import make_common_banks
from steer_opd.compass_scoring import AlignedMetricAdapter, TextScoringModel
from steer_opd.direction_bank import bank_vectors
from steer_opd.resumable_core import ROOT, atomic_json, atomic_torch, digest, score_hidden
from steer_opd.rollout_topk import commit_topk_response
from steer_opd.static_masks import freeze_static_mask, validate_static_mask
from steer_opd.static_screen import score_batch, score_native_candidate


class ToyCodec:
    def __init__(self, raw):
        self.raw = raw
        self.eos = self.special = {0}
        self.mapping_sha256 = digest({k: v.hex() for k, v in raw.items()})
        self.valid_ids = sorted(raw)
        self.support_size = max(raw)+1
        self.support_id = digest(self.valid_ids)

    def bytes(self, ids):
        return b"".join(self.raw[i] for i in ids)

    def encode(self, text):
        remaining, result = text.encode(), []
        while remaining:
            candidates = [(len(v), k) for k, v in self.raw.items() if k not in self.special and remaining.startswith(v)]
            if not candidates:
                raise ValueError("Toy text unencodable")
            n, token = max(candidates)
            result.append(token); remaining = remaining[n:]
        return result

    def event(self, token):
        token = int(token)
        return "EOS" if token == 0 else "bytes:"+self.raw[token].hex() if token in self.raw else None


def tiny(seed=1):
    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=47, hidden_size=16, intermediate_size=32, num_hidden_layers=3,
                      num_attention_heads=2, num_key_value_heads=1, head_dim=8, attention_dropout=0.)
    cfg._attn_implementation = "eager"
    return Qwen3ForCausalLM(cfg).eval()


def test_byte_decoder_covers_all_bytes_and_space():
    table = _byte_decoder()
    assert set(table.values()) == set(range(256)) and len(table) == 256
    assert table['Ġ'] == 32 and table['Ċ'] == 10


def test_teacher_student_span_mean_likelihood_not_sum_or_string_probability():
    s = ToyCodec({0: b"<eos>", 1: b"a", 2: b"b", 3: b"c"})
    t = ToyCodec({0: b"<eos>", 11: b"ab", 12: b"c"})
    plan = align_response([1, 2, 3, 0], s, {"teacher": t})
    assert plan["units"][0]["ranges"] == {"student": [0, 2], "teacher": [0, 1]}
    assert plan['schema']==2 and plan['alignment_roles']==['student','teacher']
    assert plan["alignment_mask"] == [True, True, True, False]
    def record(lps):
        return {"sampled_logps": torch.tensor(lps), "ids": torch.zeros(len(lps), 16, dtype=torch.int32), "logps": torch.zeros(len(lps), 16)}
    scores = span_scores(plan, {"student": record([-1., -2., -3., -4.]), "teacher": record([-.4, -.8])})
    assert scores["unit_mean_logps"][0]["student"] == pytest.approx(-1.5)
    assert scores['unit_mean_logps'][0]['teacher']==pytest.approx(-.4)
    assert set(scores['broadcast_mean_logps'])=={'teacher','student'}
    assert 'teacher_family_shift' not in scores
    assert torch.isnan(scores['broadcast_mean_logps']['student'][-1])


def test_utf8_split_tokens_form_one_safe_unit():
    s = ToyCodec({0: b"<eos>", 1: b"\xe4", 2: b"\xb8", 3: b"\xad"})
    t = ToyCodec({0: b"<eos>", 11: '中'.encode()})
    plan = align_response([1, 2, 3, 0], s, {"teacher": t})
    assert plan["response_text"] == '中' and len(plan["units"]) == 1
    assert plan["units"][0]["ranges"]["student"] == [0, 3]


@pytest.fixture(scope="module")
def real_codecs():
    return {role: ByteLevelCodec(AutoTokenizer.from_pretrained(ROOT / "assets/models" / name, local_files_only=True))
            for role, name in [("student", "Qwen3-4B"), ("teacher", "Qwen3.5-0.8B")]}


@pytest.mark.parametrize("text", [' hello world', '中文🙂 café', '1+1=2\n', 'é ㍿test'])
def test_real_qwen_tokenizers_keep_exact_generated_bytes_and_own_ids(real_codecs, text):
    s = real_codecs["student"]
    ids = s.encode(text)
    generated = s.bytes(ids)
    plan = align_response(ids+[s.tokenizer.eos_token_id], s, {r: real_codecs[r] for r in ("teacher",)})
    assert hashlib.sha256(generated).hexdigest() == plan["response_utf8_sha256"]
    assert plan["native_response_ids"]["student"] == ids
    assert all(real_codecs[r].bytes(v) == generated for r, v in plan["native_response_ids"].items())
    assert not plan["alignment_mask"][-1]


def test_actual_noncanonical_student_tokenization_is_preserved(real_codecs):
    s = real_codecs["student"]
    by_byte = {v: k for k, v in s.raw.items() if len(v) == 1 and k not in s.special}
    ids = [by_byte[bytes([c])] for c in b'hello']
    plan = align_response(ids, s, {"teacher": real_codecs["teacher"]})
    assert plan["native_response_ids"]["student"] == ids
    assert len(plan["native_response_ids"]["teacher"]) < len(ids)
    assert plan["units"][0]["ranges"]["student"] == [0, 5]


def test_missing_exact_bytes_are_rejected_instead_of_normalized_or_replaced():
    s = ToyCodec({0: b"<eos>", 1: b"\xff"})
    with pytest.raises(UnicodeError):
        align_response([1], s, {"teacher": s})


def test_zero_directions_are_shared_and_never_fall_back_to_other_questions():
    pooled = {qid: {"usable": True, "layers": {"2": {"pools": [torch.ones(16)*v, torch.ones(16)*v],
        "norms": [torch.ones(3), torch.ones(4)]}}} for qid, v in [('q1', 1), ('q2', 2)]}
    banks, _ = make_common_banks(pooled, [2], {"checkpoint": "S0", "step": 0})
    assert len(banks) == 2
    for bank in banks.values():
        assert not bank_vectors(bank, 'q1')[0][2].any()
        assert bank["valid_ids"] == ['q1', 'q2']


def test_cross_tokenizer_all_four_paths_keep_frozen_masks_and_score_only_aligned_units(tmp_path, monkeypatch):
    raw = {i: bytes([33+i]) for i in range(47)}; raw[0] = b"<eos>"
    sraw, traw = {**raw, 3: b"c", 4: b"d", 5: b"e"}, {**raw, 13: b"c", 14: b"d", 15: b"e"}
    s, t = ToyCodec(sraw), ToyCodec(traw)
    rows = [{"id": 'q1', "key": 'r1', "prompt_token_ids": [1, 2], "token_ids": [3, 4, 5, 0], "steering": False},
            {"id": 'q2', "key": 'r2', "prompt_token_ids": [1, 2], "token_ids": [4, 5, 0], "steering": False}]
    plans = {r['key']: align_response(r['token_ids'], s, {'teacher': t}) for r in rows}
    native_rows = [{**r, 'token_ids': plans[r['key']]['native_response_ids']['teacher']} for r in rows]
    teachers = {r['key']: v for r, v in zip(rows, score_batch(tiny(4), native_rows))}
    source = {r['key']: v for r, v in zip(rows, score_batch(tiny(3), rows))}
    adapter = AlignedMetricAdapter(plans, s, t)
    identity = {'student_checkpoint': 'S0', 'teacher_checkpoint': 'T', 'student_tokenizer_sha256': s.mapping_sha256,
        'teacher_tokenizer_sha256': t.mapping_sha256, 'support_id': 'separate_native_supports', 'metric': METRIC,
        'student_score_sha256': 's', 'teacher_score_sha256': 't'}
    masks = {r['key']: freeze_static_mask(r, adapter(r['key'], source[r['key']], teachers[r['key']])['overlap'], .5,
        identity, alignment_mask=plans[r['key']]['alignment_mask']) for r in rows}
    original = copy.deepcopy(masks)
    pooled = {qid: {'usable': True, 'layers': {'2': {'pools': [torch.arange(16).float()+i, torch.ones(16)*i],
        'norms': [torch.ones(2)*10]*2}}} for i, qid in enumerate(['q1', 'q2'], 1)}
    banks, _ = make_common_banks(pooled, [2], {'checkpoint': 'S0', 'step': 0})
    model = tiny(3)
    baseline = score_native_candidate(model, rows, teachers, masks, next(iter(banks.values())), None, metric_adapter=adapter)
    count = 0
    for direction in ('T-S',):
        for granularity in ('global', 'per_question'):
            for gate in ('all', 'static_overlap'):
                condition = {'direction': direction, 'granularity': granularity, 'gate': gate, 'layers': [2], 'operator': 'slerp', 'alpha': .1}
                result = score_native_candidate(model, rows, teachers, masks, banks[f'{direction}_{granularity}'], condition, baseline=baseline, metric_adapter=adapter)
                assert all(v['metric'] == METRIC and v['parameter_updates'] == 0 for v in result.values())
                assert all(torch.isnan(v['overlap'][-1]) and not v['alignment_mask'][-1] for v in result.values())
                assert all('teacher_family_shift' not in v for v in result.values())
                count += 1
    assert count == 4 and masks == original
    from steer_opd import compass_scoring, resumable_core
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    module = importlib.import_module('run_overlap_ablation')
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(resumable_core, 'require_gpu', lambda: None)
    monkeypatch.setattr(compass_scoring, 'load_text_scorer', lambda *args: model)
    condition = {'id': 'one', 'bank': 'T-S_global', 'direction': 'T-S', 'granularity': 'global',
                 'gate': 'all', 'layers': [2], 'operator': 'slerp', 'alpha': .1}
    job = {'student': 'S0', 'support_size': 47, 'output': 'out', 'rows': 'rows.json',
           'conditions': [condition], 'identity': {'metric': METRIC}}
    atomic_json(tmp_path / 'rows.json', rows)
    atomic_json(tmp_path / 'out/manifest.json', {'synthetic_cpu_contract_fixture': True})
    wrapped = compass_scoring.AlignedTeacherRecords(teachers, adapter)
    module.run(job, rows, masks, wrapped, banks)
    summary = json.loads((tmp_path / 'out/summaries.json').read_text())['one']
    assert summary['positions'] == 5 and summary['metric'] == METRIC
    def forbid(*args):
        raise AssertionError('Cached aligned scores must not load a model')
    monkeypatch.setattr(resumable_core, 'require_gpu', forbid)
    monkeypatch.setattr(compass_scoring, 'load_text_scorer', forbid)
    module.run(job, rows, masks, wrapped, banks)


def test_text_wrapper_and_native_valid_support_mask_keep_probabilities_finite():
    model = TextScoringModel(tiny())
    model._opd_vocab_size = 47
    model._opd_valid_output_ids = list(range(46))
    row = {'prompt_token_ids': [1, 2], 'token_ids': [3, 4]}
    record = score_batch(model, [row])[0]
    assert not (record['ids'] == 46).any() and torch.isfinite(record['sampled_logps']).all()
    assert torch.isfinite(record['logps']).all() and record['ids'].shape == (2,16)


def test_alignment_mask_preserves_missing_values_and_excludes_eos_from_gate():
    row = {'id': 'q', 'prompt_token_ids': [1], 'token_ids': [2, 0], 'steering': False}
    identity = {k: k for k in ('student_checkpoint', 'teacher_checkpoint', 'student_tokenizer_sha256', 'teacher_tokenizer_sha256', 'support_id', 'student_score_sha256', 'teacher_score_sha256')}
    identity['metric'] = METRIC
    mask = freeze_static_mask(row, [0., None], .5, identity, alignment_mask=[True, False])
    assert mask['source_overlap'] == [0., None] and mask['requested_mask'] == [True, False]
    assert validate_static_mask(mask, row) == [True, False]


def test_read_only_shared_model_links_are_allowed_but_artifact_escape_is_rejected(tmp_path, monkeypatch):
    from steer_opd import compass_job
    monkeypatch.setattr(compass_job, 'ROOT', tmp_path / 'project')
    assets = compass_job.ROOT / 'assets/models'; assets.mkdir(parents=True)
    shared = tmp_path / 'shared'; shared.mkdir()
    (assets / 'model').symlink_to(shared)
    assert compass_job.model_path('assets/models/model').resolve() == shared
    with pytest.raises(ValueError, match='project'):
        compass_job.project_path('assets/models/model')


def test_real_tokenizers_complete_source_cache_to_static_mask_chain(tmp_path, monkeypatch, real_codecs):
    from steer_opd import compass_job
    from steer_opd.math_data import prompt_ids
    monkeypatch.setattr(compass_job, 'ROOT', tmp_path)
    questions = [{'id': 'q1', 'question': 'What is one plus one?', 'answer': '2'}]
    atomic_json(tmp_path / 'questions.json', questions)
    s = real_codecs['student']
    response = s.encode('hello world')+[s.tokenizer.eos_token_id]
    row = {'id': 'q1', 'key': 'r1', 'prompt_token_ids': prompt_ids(s.tokenizer, questions[0]['question'], enable_thinking=False),
           'token_ids': response, 'model': 'S0', 'steering': False, 'sampled_logps': [-8.]*len(response)}
    n = len(response)
    source = {'ids': torch.arange(16, dtype=torch.int32).repeat(n, 1),
        'logps': (-4.-torch.arange(16)*.1).repeat(n, 1), 'sampled_logps': torch.full((n,), -8.),
        'source': 'vllm_rollout_raw_logprobs', 'support_size': s.support_size, 'k': 16, 'probability_mode': 'raw_logprobs'}
    row = commit_topk_response(tmp_path / 'source', row, source)
    atomic_json(tmp_path / 'rows.json', [row])
    pooled = {'q1': {'usable': True, 'layers': {'26': {'pools': [torch.arange(16).float(), torch.zeros(16)],
                                                        'norms': [torch.ones(2)*10]*2}}}}
    banks, _ = make_common_banks(pooled, [26], {'checkpoint': 'S0', 'step': 0})
    paths, conditions = {}, []
    for name, bank in banks.items():
        paths[name] = f'banks/{name}.pt'; atomic_torch(tmp_path / paths[name], bank)
        for gate in ['all', 'static_overlap']:
            conditions.append({'id': f'{name}_{gate}', 'bank': name, 'direction': bank['method']['direction'],
                'granularity': bank['method']['granularity'], 'gate': gate, 'layers': [26], 'operator': 'slerp', 'alpha': .1})
    job = {'stage': 'static_overlap_screen', 'optimizer_updates': 0, 'student': 'S0', 'teacher': 'T',
           'identity': {'metric': METRIC}, 'rows': 'rows.json', 'questions': 'questions.json', 'alignment_dir': 'align',
           'source_dir': 'source', 'support_size': s.support_size, 'teacher_scores': 'T_scores',
           'banks': paths, 'conditions': conditions, 'threshold': .6, 'output': 'result'}
    _, _, native_rows, _ = compass_job.prepare_plans(job, real_codecs)
    for role in ['teacher']:
        c = real_codecs[role]
        native_row = native_rows[role][0]; length = len(native_row['token_ids'])
        binding = compass_job.native_score_binding(job, role, native_row, c)
        score = {'ids': torch.arange(8, 24, dtype=torch.int32).repeat(length, 1),
                 'logps': (-4.-torch.arange(16)*.1).repeat(length, 1), 'sampled_logps': torch.full((length,), -6.), 'binding': binding}
        atomic_torch(tmp_path / job[f'{role}_scores'] / 'r1.pt', score)
    prepared = compass_job.prepare_screen(job, real_codecs)
    assert len(prepared[3]) == 2 and len(job['conditions']) == 4
    artifact = prepared[1]['r1']
    assert artifact['schema'] == 2 and artifact['alignment_mask'][-1] is False
    assert artifact['requested_mask'][-1] is False and artifact['source_overlap'][-1] is None
    assert artifact['identity']['student_tokenizer_sha256'] != artifact['identity']['teacher_tokenizer_sha256']
    assert artifact['identity']['metric_version'] == 4
    assert artifact['identity']['matching_rule_version'] == 'native_token_prefix_compatibility_v1'
    assert compass_job.prepare_screen(job, real_codecs)[1] == prepared[1]
    native = torch.load(tmp_path / 'T_scores/r1.pt', weights_only=True)
    native['binding'] = digest({'row':native_rows['teacher'][0], 'checkpoint':job['teacher'],
                               'tokenizer':real_codecs['teacher'].mapping_sha256,
                               'support_id':real_codecs['teacher'].support_id})
    atomic_torch(tmp_path / 'T_scores/r1.pt', native)
    with pytest.raises(ValueError, match='cache has changed'):
        compass_job.prepare_screen(job, real_codecs)


@pytest.mark.parametrize('metadata', ['assets/models/Qwen3.5-0.8B', 'assets/metadata/Qwen3.5-35B-A3B'])
def test_actual_qwen35_dense_and_moe_text_architectures_score_on_cpu(metadata, monkeypatch):
    from transformers import AutoConfig, AutoModel
    original = AutoConfig.from_pretrained(ROOT / metadata, local_files_only=True).text_config
    values = original.to_dict()
    values.update(vocab_size=47, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                  num_attention_heads=2, num_key_value_heads=1, head_dim=16, max_position_embeddings=64,
                  linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=8, linear_value_head_dim=8,
                  layer_types=['linear_attention', 'full_attention'], num_experts=4, num_experts_per_tok=2,
                  moe_intermediate_size=16, shared_expert_intermediate_size=16,
                  rope_parameters={'rope_type': 'default', 'rope_theta': 10000000, 'mrope_section': [2, 3, 3],
                                   'mrope_interleaved': True, 'partial_rotary_factor': 1.0})
    cfg = type(original).from_dict(values)
    cfg._attn_implementation = 'eager'
    decoder = AutoModel.from_config(cfg).eval()
    # Installed FLA is a GPU-only package. Exercise the actual Transformers
    # Torch reference functions on CPU, rather than invoking a Triton driver.
    import inspect
    modeling = importlib.import_module(type(decoder).__module__)
    for name in ('torch_chunk_gated_delta_rule', 'torch_recurrent_gated_delta_rule', 'causal_conv1d_fn', 'causal_conv1d_update'):
        monkeypatch.setattr(modeling, name, inspect.unwrap(getattr(modeling, name)))
    owner = torch.nn.Module(); owner.model = torch.nn.Module(); owner.model.language_model = decoder
    owner.lm_head = torch.nn.Linear(32, 47, bias=False); owner.config = SimpleNamespace(text_config=cfg)
    model = TextScoringModel(owner).eval()
    rows = [{'prompt_token_ids': [1, 2], 'token_ids': [3, 4, 5]}, {'prompt_token_ids': [1], 'token_ids': [2]}]
    scores = score_batch(model, rows)
    assert [len(s['sampled_logps']) for s in scores] == [3, 1]
    assert all(torch.isfinite(s['sampled_logps']).all() and torch.isfinite(s['logps']).all() for s in scores)
    single = score_batch(model, rows[:1])[0]
    torch.testing.assert_close(scores[0]['logps'], single['logps'], atol=1e-5, rtol=1e-5)


def test_campaign_builds_fixed_and_annealed_candidates_and_native_generation_profiles(real_codecs):
    from steer_opd.compass_campaign import generation_job, make_base_job, validate_config
    cfg = json.loads((ROOT / 'configs/ablation.json').read_text())
    validate_config(cfg)
    questions = [{'id': 'q1', 'question': 'What is 1+1?', 'answer': '2'}]
    source = generation_job(cfg, real_codecs['student'], 'student', questions, source=True)
    teacher = generation_job(cfg, real_codecs['teacher'], 'teacher', questions)
    for stage in (source, teacher):
        assert stage['execution_hold'] == cfg['execution_hold']
        assert stage['execution_blockers'] == cfg['execution_blockers']
        assert 'max_num_seqs' not in stage and 'batch_size' not in stage
    assert source['save_rollout_top_k'] == 16 and source['temperature'] == 1 and source['tensor_parallel_size'] == 1
    assert teacher['tensor_parallel_size'] == 2 and teacher['support_size'] != source['support_size']
    job = make_base_job(cfg, real_codecs, 'rows.json')
    assert job['execution_hold'] == cfg['execution_hold']
    assert job['backend'] == 'easysteer'
    assert len(job['conditions']) == 4 and len(job['banks']) == 2
    assert all(c['layers'] == [26] and c['alpha'] == .1 for c in job['conditions'])
    with pytest.raises(ValueError, match='zero updates'):
        validate_config({**cfg, 'optimizer_updates': 1})
    for phase in ('param','baseline'):
        with pytest.raises(ValueError,match='Only the overlap phase'):
            validate_config({**cfg,'phase':phase})


def test_plotting_keeps_requested_first_three_and_marks_unscored_positions(tmp_path):
    from steer_opd.static_plotting import plot_comparisons
    rows = [{'id': 'q', 'key': key} for key in ['r1', 'r2', 'r3']]
    baseline = {key: {'overlap': torch.tensor([.5, float('nan')])} for key in ['r1', 'r2', 'r3']}
    candidate = {key: {'overlap': torch.tensor([.6, float('nan')]), 'steering_mask': torch.tensor([True, True]),
        'requested_mask': torch.tensor([True, True]), 'alignment_mask': torch.tensor([True, False]),
        'binding': key, 'mask_id': key, 'metric': METRIC} for key in ['r1', 'r2', 'r3']}
    plot_comparisons(rows, baseline, candidate, tmp_path, requested_keys=['unscorable_original_first', 'r1', 'r2'])
    manifest = json.loads((tmp_path / 'manifest.json').read_text())
    assert manifest['unavailable_keys'] == ['unscorable_original_first']
    assert [row['key'] for row in manifest['rows']] == ['r1', 'r2']
    assert not (tmp_path / 'r3.png').exists()
    assert all(row['metric_valid_positions'] == 1 and row['steered'] == 2 for row in manifest['rows'])
