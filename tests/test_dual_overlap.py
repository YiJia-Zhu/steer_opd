"""Raw Top16 metrics, capture and ordinary vLLM resume contracts on CPU."""
import copy
import csv
import importlib
import json
from types import SimpleNamespace as NS

import pytest
import torch

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


def test_aligned_dual_metrics_broadcast_common_start_and_leave_terminal_missing():
    s, t = codec(), codec(True)
    plan = align_response([1, 4, 0], s, {'teacher': t})
    def record(c, other, n):
        return {'ids': torch.arange(16).repeat(n, 1), 'logps': torch.full((n, 16), -5.),
                'sampled_logps': torch.full((n,), -5.)}
    a, b = record(s, t, 3), record(t, s, len(plan['native_response_ids']['teacher']))
    metrics = aligned_metrics(plan, a, b, s, t)
    assert metrics['P_overlap_available'] and torch.isnan(metrics['P_overlap'][-1])
    assert torch.isfinite(metrics['P_overlap'][:-1]).all()
    adapter = AlignedMetricAdapter({'r': plan}, s, t, require_dual=True)
    adapter('r', a, b)
    with pytest.raises(ValueError, match='alignment plan'):
        adapter('r', {**a, 'ids': a['ids'][:, :15], 'logps': a['logps'][:, :15]}, b)


def test_capture_pool_preserves_raw_body_means_not_prediction_rows(monkeypatch):
    from steer_opd import easysteer_extraction as ex
    monkeypatch.setattr(ex, 'encode_answer', lambda *args, **kwargs: {
        'prompt_ids':[1,2],'response_ids':[3,4,5],'body_mask':[True]*3,'offsets':[(0,1),(1,2),(2,3)],
        'original_response_tokens':20,'retained_response_tokens':3,'max_context_len':5,
        'context_truncated':True,'pooling':'sv_training_all_retained_answer_tokens'})
    states = {26: torch.tensor([[100., 100.], [90., 90.], [1., 3.], [5., 7.]])}
    result = ex.pooled_capture(None, {'student_prompt_text': 'prompt'}, [{'text': 'abc','finish_reason':'length'}]*2,
                               [states]*2, [26])
    assert result['usable']
    torch.testing.assert_close(result['layers']['26']['pools'][0], torch.tensor([32., 100./3]))
    torch.testing.assert_close(result['layers']['26']['norms'][0],states[26][1:].norm(dim=-1))
    assert set(result)=={'usable','layers'}


@pytest.mark.parametrize('gate', ['overlap', 'P_overlap'])
def test_dual_complete_chain_plots_summary_and_resume_without_model(tmp_path, monkeypatch, gate):
    from steer_opd import compass_scoring, resumable_core
    from steer_opd.compass_banks import make_common_banks
    from steer_opd.static_screen import score_batch
    s, t = codec(), codec(True)
    rows = [{'id':'q', 'key':'r', 'prompt_token_ids':[1,2], 'token_ids':[1,4,0], 'steering':False}]
    plan = align_response(rows[0]['token_ids'], s, {'teacher':t})
    model = tiny(3)
    native_model = tiny(4)
    native_row = {**rows[0], 'token_ids':plan['native_response_ids']['teacher']}
    teacher = score_batch(native_model, [native_row])[0]
    adapter = AlignedMetricAdapter({'r':plan}, s, t, require_dual=True)
    source = score_batch(model, rows)[0]; metrics = adapter('r', source, teacher)
    identity = {k:k for k in ('student_checkpoint','teacher_checkpoint','student_tokenizer_sha256',
        'teacher_tokenizer_sha256','support_id','student_score_sha256','teacher_score_sha256')}
    identity.update(student_checkpoint='S0', teacher_checkpoint='T', metric=METRIC, gate_metric=gate)
    mask = freeze_static_mask(rows[0], metrics[gate], .8, identity,
        alignment_mask=plan['alignment_mask'], source_metrics={k:metrics[k] for k in ['overlap','P_overlap']})
    assert validate_static_mask(mask, rows[0])[-1] is False
    original = copy.deepcopy(mask)
    pool = {'q': {'usable':True, 'layers':{'2':{'pools':[torch.arange(16).float(),torch.zeros(16)],
                                                 'norms':[torch.ones(2)*10]*2}}}}
    banks, _ = make_common_banks(pool, [2], {'checkpoint':'S0','step':0})
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    module = importlib.import_module('run_overlap_ablation')
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


def test_top16_cache_hash_detects_probability_and_protocol_changes():
    from steer_opd.rollout_topk import tensor_hash
    record = {'ids':torch.arange(16).reshape(1,16), 'logps':torch.full((1,16),-5.),
              'sampled_logps':torch.tensor([-5.]), 'source':'vllm_rollout_raw_logprobs',
              'support_size':47,'probability_mode':'raw_logprobs'}
    before = tensor_hash(record)
    record['logps'][0,15] -= .1
    assert tensor_hash(record) != before
    before = tensor_hash(record)
    record['support_size'] = 48
    assert tensor_hash(record) != before


def test_native_source_generation_commits_only_top16_and_resumes_without_engine(tmp_path, monkeypatch):
    import sys
    from types import ModuleType
    from steer_opd.rollout_topk import load_rollout_topk
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    module = importlib.import_module('generate_static_sources')
    monkeypatch.setattr(module, 'project_path', lambda p: tmp_path/p)
    c = codec(); monkeypatch.setattr(module, 'load_codecs', lambda _: {'student':c, 'teacher':c})
    vllm = ModuleType('vllm'); vllm.SamplingParams = lambda **kwargs: NS(**kwargs)
    params = ModuleType('vllm.sampling_params'); params.RequestOutputKind = NS(FINAL_ONLY='final')
    monkeypatch.setitem(sys.modules,'vllm',vllm); monkeypatch.setitem(sys.modules,'vllm.sampling_params',params)
    requests = [{'id':'q', 'key':'q_0','prompt_token_ids':[7,8,9],'seed':42}]
    lp = {i:NS(logprob=-5.-.01*i,rank=i+1) for i in range(16)}
    sample = NS(token_ids=[1],text='a',finish_reason='stop',stop_reason=0,logprobs=[lp])
    output = NS(request_id='0-uuid',finished=True,outputs=[sample],prompt_token_ids=[7,8,9])
    class Engine:
        def generate(self,prompts,options,**kwargs):
            assert len(prompts)==len(requests) and options[0].logprobs==16
            assert options[0].presence_penalty==2.
            assert kwargs['steering'] is False
            return [output]
        def collective_rpc(self,name,args):
            pytest.fail('Generation must use ordinary vLLM logprobs without a probability RPC')
    calls=[]
    def create(*args,**kwargs): calls.append(kwargs); return Engine()
    monkeypatch.setattr(module,'create_engine',create)
    job = {'output':'source','role':'student','model':'S0','save_rollout_top_k':16,'support_size':47,
           'requests':requests,'max_tokens':4,'temperature':1.,'top_p':1.,'gpu_memory_utilization':.75,
           'presence_penalty':2.}
    module.generate(job)
    row=json.loads((tmp_path/'source/q_0.json').read_text())
    score=load_rollout_topk(tmp_path/'source',row,16,47)
    assert score['ids'].shape == score['logps'].shape == (1,16) and row['steering_mask']==[False]
    assert not any('probs' in name for name in score)
    assert not (tmp_path/'source/projection.pt').exists()
    module.generate(job)
    assert (tmp_path/'source/complete.json').is_file()
    assert len(calls)==1  # Reuse the completed sparse source cache.


def test_native_support_crops_qwen3_and_nested_qwen35_heads():
    from steer_opd.easysteer_static import crop_native_support
    for nested, native, padded in [(False,151669,151936),(True,248077,248320)]:
        head=NS(logits_processor=NS(org_vocab_size=padded))
        model=NS(language_model=head) if nested else head
        crop_native_support(model,native)
        assert head.logits_processor.org_vocab_size==native
        with pytest.raises(ValueError,match='support'):
            crop_native_support(model,padded+1)
