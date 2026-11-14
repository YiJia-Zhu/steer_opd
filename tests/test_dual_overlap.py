"""Full-distribution metric and pinned-runner coordinate contracts on CPU."""
import copy
import csv
import importlib
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from steer_opd.byte_probabilities import (BYTE_CLASSES, projection_spec, probability_overlap,
    summarize_logits, validate_byte_probabilities, validate_projection)
from steer_opd.engine_projection import ProjectionCollector, install_projection, select_prediction_statistics
from steer_opd.compass_alignment import METRIC, align_response, aligned_metrics
from steer_opd.compass_scoring import AlignedMetricAdapter, AlignedTeacherRecords
from steer_opd.static_masks import freeze_static_mask, validate_static_mask
from steer_opd.resumable_core import ROOT, atomic_json
from test_compass_alignment import ToyCodec, tiny


def codec(teacher=False):
    raw = {i: bytes([65+i]) for i in range(47)}
    raw.update({0: b'<eos>', 1: b'a', 2: b'ab', 3: b'ac', 4: b'b', 46: b'<ctl>'})
    if teacher:
        raw[2] = b'az'
    result = ToyCodec(raw)
    result.special = {0, 46}
    return result


def test_full_mass_first_byte_eos_controls_and_undefined_padded_rows():
    c = codec(); c.raw.pop(45); c.valid_ids.remove(45)
    spec = projection_spec(c, codec(True))
    logits = torch.full((1, 49), -torch.inf)
    # Large padded/undefined logits must not enter the denominator.
    logits[0, [1, 2, 3, 0, 46, 45, 47, 48]] = torch.tensor([0., 0., 0., 0., 0., 100., 100., 100.])
    result = summarize_logits(logits, spec)
    assert result['byte_probs'][0, ord('a')] == pytest.approx(.6)
    assert result['byte_probs'][0, 256] == pytest.approx(.2)
    assert result['byte_probs'][0, 257] == pytest.approx(.2)
    assert result['entropy'].item() == pytest.approx(np.log(5))
    validate_projection(result, spec, 1)
    validate_byte_probabilities(result['byte_probs'].double())
    damaged = {**result, 'projection_id': 'wrong'}
    with pytest.raises(ValueError, match='identity'):
        validate_projection(damaged, spec, 1)
    with pytest.raises(ValueError, match='normalized'):
        validate_byte_probabilities(result['byte_probs']*.9)


def test_probability_overlap_has_full_tail_and_different_tokenization_can_agree():
    s, t = codec(), codec(True)
    # No same full token bytes; both distributions still predict next byte a.
    a = torch.full((1, 47), -torch.inf); a[0, 2] = 0
    b = torch.full((1, 47), -torch.inf); b[0, 2] = 0
    sa, tb = summarize_logits(a, projection_spec(s, t)), summarize_logits(b, projection_spec(t, s))
    assert probability_overlap(sa['byte_probs'], tb['byte_probs']).item() == 1
    # Full tail matters even when Top16 is unchanged.
    p = torch.zeros(1, BYTE_CLASSES); p[0, [97, 98]] = torch.tensor([.7, .3])
    q = torch.zeros_like(p); q[0, [97, 99]] = torch.tensor([.7, .3])
    assert probability_overlap(p, q).item() == pytest.approx(.7)
    assert probability_overlap(p, p).item() == 1
    assert probability_overlap(p, q).item() == probability_overlap(q, p).item()


def test_aligned_dual_metrics_broadcast_common_start_and_leave_terminal_missing():
    s, t = codec(), codec(True)
    plan = align_response([1, 4, 0], s, {'teacher': t, 'reference': t})
    def record(c, other, n):
        score = summarize_logits(torch.zeros(n, 47), projection_spec(c, other))
        return {**score, 'ids': torch.arange(16).repeat(n, 1), 'logps': torch.full((n, 16), -5.),
                'sampled_logps': torch.full((n,), -5.)}
    a, b = record(s, t, 3), record(t, s, len(plan['native_response_ids']['teacher']))
    metrics = aligned_metrics(plan, a, b, s, t)
    assert metrics['P_overlap_available'] and torch.isnan(metrics['P_overlap'][-1])
    assert torch.isfinite(metrics['P_overlap'][:-1]).all()
    adapter = AlignedMetricAdapter({'r': plan}, s, t, {'r': b}, require_dual=True)
    adapter('r', a, b)
    with pytest.raises(ValueError, match='identity'):
        adapter('r', {**a, 'projection_id': 'old_vocab'}, b)


def fake_runner():
    def prompt(logits_fn, hidden, batch, *unused):
        return torch.cat([logits_fn(part) for part in hidden.split(2)])
    model = NS(compute_logits=lambda hidden: hidden)
    def sample(hidden, batch, grammar):
        return model.compute_logits(hidden[batch.logits_indices])
    return NS(prompt_logprobs_worker=NS(compute_prompt_logprobs=prompt), sample=sample,
              model=model, batch_sharder=None)


def test_v2_prompt_chunks_interleaved_requests_and_decode_positions():
    spec = projection_spec(codec(), codec(True))
    runner = fake_runner(); collector = install_projection(runner, spec)
    logits = torch.randn(5, 47, generator=torch.Generator().manual_seed(8))
    batch = NS(req_ids=['b-uuid', 'a-uuid'], query_start_loc_np=np.array([0, 3, 5]),
               num_computed_tokens_np=np.array([4, 10]))
    observed = runner.prompt_logprobs_worker.compute_prompt_logprobs(lambda x: x, logits, batch, None, None, None)
    torch.testing.assert_close(observed, logits)  # Hook never changes the model output.
    b = collector.pop('b'); a = collector.pop('a-uuid')
    assert b['positions'] == [4, 5, 6] and a['positions'] == [10, 11]
    expected = summarize_logits(logits, spec)['byte_probs']
    torch.testing.assert_close(b['byte_probs'], expected[:3])
    torch.testing.assert_close(a['byte_probs'], expected[3:])
    # Decode request order can change independently of previous prefill order.
    decode = NS(req_ids=['a-uuid', 'b-uuid'], positions=torch.tensor([12, 7]),
                logits_indices=torch.tensor([0, 1]), cu_num_logits_np=np.array([0, 1, 2]), num_draft_tokens=0)
    torch.testing.assert_close(runner.sample(logits[:2], decode, None), logits[:2])
    assert collector.pop('a')['positions'] == [12]
    assert collector.pop('b')['positions'] == [7]
    decode.num_draft_tokens = 1
    with pytest.raises(RuntimeError, match='Speculative'):
        runner.sample(logits[:2], decode, None)


def test_collector_recompute_consistency_missing_rows_and_tp_mismatch():
    spec = projection_spec(codec())
    collect = ProjectionCollector(spec)
    logits = torch.zeros(2, 47)
    collect.add('0-uuid', [2, 3], logits)
    collect.add('0-uuid', [3], logits[1:])
    different = logits[1:].clone(); different[0, 4] = 3
    with pytest.raises(ValueError, match='Repeated'):
        collect.add('0-uuid', [3], different)
    record = collect.pop('0')
    row = {'prompt_token_ids': [1, 2, 3], 'token_ids': [4, 5]}
    selected = select_prediction_statistics([None, record, record], row)
    assert len(selected['byte_probs']) == 2
    with pytest.raises(ValueError, match='missing'):
        select_prediction_statistics([record], {**row, 'token_ids': [4, 5, 6]})
    corrupt = copy.deepcopy(record); corrupt['common_vocab_mass'] += .1
    with pytest.raises(ValueError, match='parallel'):
        select_prediction_statistics([record, corrupt], row)


def test_capture_pool_preserves_raw_body_means_not_prediction_rows(monkeypatch):
    from steer_opd import easysteer_extraction as ex
    monkeypatch.setattr(ex, 'encode_content', lambda *args, **kwargs: ([3, 4, 5], [False, True, True], [(0,1),(1,2),(2,3)]))
    states = {26: torch.tensor([[100., 100.], [90., 90.], [1., 3.], [5., 7.]])}
    saved = []
    result = ex.pooled_capture(None, {'student_prompt_ids': [1, 2]}, [{'text': 'abc'}]*3,
                               [states]*3, [26], lambda i, payload: saved.append(payload))
    assert result['usable']
    torch.testing.assert_close(result['layers']['26']['pools'][0], torch.tensor([3., 5.]))
    assert saved[0]['state_input_positions'].tolist() == [1, 2, 3, 4]
    assert saved[0]['prediction_state_rows'].tolist() == [0, 1, 2]
    assert saved[0]['states']['26'].dtype == torch.bfloat16


@pytest.mark.parametrize('gate', ['overlap', 'P_overlap'])
def test_dual_complete_chain_plots_summary_and_resume_without_model(tmp_path, monkeypatch, gate):
    from steer_opd import compass_scoring, resumable_core
    from steer_opd.compass_banks import make_common_banks
    from steer_opd.static_screen import score_batch
    s, t = codec(), codec(True)
    rows = [{'id':'q', 'key':'r', 'prompt_token_ids':[1,2], 'token_ids':[1,4,0], 'steering':False}]
    plan = align_response(rows[0]['token_ids'], s, {'teacher':t, 'reference':t})
    model = tiny(3); model._opd_byte_projection = projection_spec(s, t)
    native_model = tiny(4); native_model._opd_byte_projection = projection_spec(t, s)
    native_row = {**rows[0], 'token_ids':plan['native_response_ids']['teacher']}
    teacher = score_batch(native_model, [native_row])[0]
    adapter = AlignedMetricAdapter({'r':plan}, s, t, {'r':teacher}, require_dual=True)
    source = score_batch(model, rows)[0]; metrics = adapter('r', source, teacher)
    identity = {k:k for k in ('student_checkpoint','teacher_checkpoint','student_tokenizer_sha256',
        'teacher_tokenizer_sha256','support_id','student_score_sha256','teacher_score_sha256')}
    identity.update(student_checkpoint='S0', teacher_checkpoint='T', metric=METRIC, gate_metric=gate)
    mask = freeze_static_mask(rows[0], metrics[gate], .8, identity,
        alignment_mask=plan['alignment_mask'], source_metrics={k:metrics[k] for k in ['overlap','P_overlap']})
    assert validate_static_mask(mask, rows[0])[-1] is False
    original = copy.deepcopy(mask)
    pool = {'q': {'usable':True, 'layers':{'2':{'pools':[torch.arange(16).float(),torch.ones(16),torch.zeros(16)],
                                                 'norms':[torch.ones(2)*10]*3}}}}
    banks, _ = make_common_banks(pool, [2], {'checkpoint':'S0','step':0})
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    module = importlib.import_module('run_static_overlap_screen')
    monkeypatch.setattr(module,'ROOT',tmp_path)
    monkeypatch.setattr(resumable_core,'require_gpu',lambda:None)
    monkeypatch.setattr(compass_scoring,'load_text_scorer',lambda *args:model)
    condition = {'id':'one','bank':'T-S_global','direction':'T-S','granularity':'global','gate':'static_overlap',
                 'layers':[2],'operator':'slerp','alpha':.1}
    job = {'student':'S0','support_size':47,'output':'out','rows':'rows.json','conditions':[condition],
           'identity':{'metric':METRIC},'require_dual_metrics':True,'gate_metric':gate,'rank_metric':gate}
    atomic_json(tmp_path/'rows.json',rows); atomic_json(tmp_path/'out/manifest.json',{'fixture':True})
    wrapped = AlignedTeacherRecords({'r':teacher},adapter)
    module.run(job,rows,{'r':mask},wrapped,banks)
    summary = json.loads((tmp_path/'out/summaries.json').read_text())['one']
    assert set(summary['metrics']) == {'overlap','P_overlap'} and summary['ranking_metric'] == gate
    assert summary['metrics']['P_overlap']['positions'] == 2
    for metric in ['overlap','P_overlap']:
        folder = tmp_path/'out/one/comparison'/metric
        assert (folder/'first3.pdf').is_file() and (folder/'r.png').is_file()
        records = list(csv.DictReader((folder/'r.csv').open()))
        assert records[-1]['metric_valid'] == '0'
        assert f'baseline_{metric}' in records[0]
    baseline = torch.load(tmp_path/'out/baseline/r.pt',weights_only=True)
    for metric in ['overlap','P_overlap']:
        torch.testing.assert_close(baseline[f'source_rollout_vs_replay_delta_{metric}'][:-1],torch.zeros(2,dtype=torch.float64))
    assert mask == original
    def forbid(*args): raise AssertionError('Committed screen must not load models')
    monkeypatch.setattr(compass_scoring,'load_text_scorer',forbid)
    monkeypatch.setattr(resumable_core,'require_gpu',forbid)
    module.run(job,rows,{'r':mask},wrapped,banks)


def test_checkpoint_contents_invalidate_same_path_and_unchanged_cache_reuses_hashes(tmp_path, monkeypatch):
    from steer_opd import checkpoint_identity as ci
    model = tmp_path/'model'; model.mkdir()
    (model/'config.json').write_text('{}')
    (model/'model.safetensors').write_bytes(b'weights-v1')
    first = ci.checkpoint_content_id(model, tmp_path/'cache')
    sha = ci.hashlib.sha256
    calls = []
    def tracked(*args, **kwargs):
        calls.append(True); return sha(*args, **kwargs)
    monkeypatch.setattr(ci.hashlib, 'sha256', tracked)
    real_time = ci.time_ns
    monkeypatch.setattr(ci, 'time_ns', lambda: real_time()+3_000_000_000)
    assert ci.checkpoint_content_id(model, tmp_path/'cache') == first
    # Only the two small identity JSON digests are computed on a cache hit.
    cached_calls = len(calls)
    monkeypatch.setattr(ci, 'time_ns', real_time)
    (model/'model.safetensors').write_bytes(b'weights-v2')
    assert ci.checkpoint_content_id(model, tmp_path/'cache') != first
    assert len(calls)-cached_calls > cached_calls


def test_upstream_fingerprint_separates_payload_slerp_layer_selector_and_normalize():
    import ast
    path = ROOT/'ref/easysteer_runtime/vllm/model_hooks/steering/request.py'
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name=='config_fingerprint')
             or (isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
                 and n.target.id in ('STEER_CLAUSE_FIELDS','STEER_APPLY_FIELDS'))]
    scope = {'SteeringRequest':NS}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)
    fingerprint = scope['config_fingerprint']
    values = dict(payload_sha256='vector_content',scale=.1,target_layers=[26],
                  apply_spec={'prompt_positions':[4,8]},algorithm='steer_opd_slerp',normalize=False)
    def request(vector, name='display'):
        return NS(vectors=[NS(**vector)],steer_vector_name=name,conflict_resolution='priority')
    base = fingerprint(request(values))
    assert base == fingerprint(request(values,name='other_label'))
    for field, replacement in [('payload_sha256','other_content'),('scale',.3),('target_layers',[25]),
        ('apply_spec',{'prompt_positions':[5,8]}),('algorithm','direct'),('normalize',True)]:
        assert base != fingerprint(request({**values,field:replacement}))


def test_full_byte_cache_hash_detects_tail_changes():
    from steer_opd.rollout_topk import tensor_hash
    record = {'ids':torch.arange(16).reshape(1,16), 'logps':torch.full((1,16),-5.),
              'sampled_logps':torch.tensor([-5.]), 'byte_probs':torch.ones(1,258)/258,
              'projection_id':'native'}
    before = tensor_hash(record)
    record['byte_probs'][0,200] += .001; record['byte_probs'][0,201] -= .001
    assert tensor_hash(record) != before


def test_native_source_generation_commits_dual_statistics_and_resumes_without_engine(tmp_path, monkeypatch):
    import sys
    from types import ModuleType
    from steer_opd.rollout_topk import load_rollout_topk
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    module = importlib.import_module('generate_static_sources')
    monkeypatch.setattr(module, 'project_path', lambda p: tmp_path/p)
    c = codec(); monkeypatch.setattr(module, 'load_codecs', lambda _: {'student':c, 'teacher':c, 'reference':c})
    vllm = ModuleType('vllm'); vllm.SamplingParams = lambda **kwargs: NS(**kwargs)
    params = ModuleType('vllm.sampling_params'); params.RequestOutputKind = NS(FINAL_ONLY='final')
    monkeypatch.setitem(sys.modules,'vllm',vllm); monkeypatch.setitem(sys.modules,'vllm.sampling_params',params)
    requests = [{'id':'q', 'key':'q_0','prompt_token_ids':[7,8,9],'seed':42}]
    lp = {i:NS(logprob=-5.-.01*i,rank=i+1) for i in range(16)}
    sample = NS(token_ids=[1],text='a',finish_reason='stop',stop_reason=0,logprobs=[lp])
    output = NS(request_id='0-uuid',finished=True,outputs=[sample],prompt_token_ids=[7,8,9])
    stats = summarize_logits(torch.zeros(1,47),projection_spec(c,c))
    class Engine:
        def generate(self,prompts,options,**kwargs):
            assert len(prompts)==len(requests) and options[0].logprobs==16
            assert kwargs['steering'] is False
            return [output]
        def collective_rpc(self,name,args):
            assert name=='project_pop_byte_statistics' and args==('0-uuid',)
            return [{**stats,'positions':[2]}]
    calls=[]
    def create(*args,**kwargs): calls.append(kwargs); return Engine()
    monkeypatch.setattr(module,'create_engine',create)
    job = {'output':'source','role':'student','model':'S0','save_rollout_top_k':16,'support_size':47,
           'requests':requests,'max_tokens':4,'temperature':1.,'top_p':1.,'gpu_memory_utilization':.75}
    module.generate(job)
    row=json.loads((tmp_path/'source/q_0.json').read_text())
    score=load_rollout_topk(tmp_path/'source',row,16,47)
    assert score['byte_probs'].shape==(1,258) and row['steering_mask']==[False]
    module.generate(job)
    assert (tmp_path/'source/complete.json').is_file()
    assert len(calls)==1  # Reuse completed source and its full-byte distribution.


def test_native_support_crops_qwen3_and_nested_qwen35_heads():
    from steer_opd.engine_projection import crop_native_support
    for nested, native, padded in [(False,151669,151936),(True,248077,248320)]:
        head=NS(logits_processor=NS(org_vocab_size=padded))
        model=NS(language_model=head) if nested else head
        crop_native_support(model,native)
        assert head.logits_processor.org_vocab_size==native
        with pytest.raises(ValueError,match='support'):
            crop_native_support(model,padded+1)
