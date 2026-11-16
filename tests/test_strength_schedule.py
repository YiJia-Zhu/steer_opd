"""Independent frozen assignment, heterogeneous whole-call scoring, and zero tail."""
import copy
import importlib
import math

import pytest
import torch

from steer_opd.compass_campaign import scheme_conditions,make_base_job
from steer_opd.strength_schedule import (LINEAR,schedule_value,freeze_schedule,validate_schedule,
    row_strength,confirm_with_counterparts)
from steer_opd.resumable_core import ROOT,atomic_json,read_json


def test_exact_requested_points_and_twenty_zero_tail():
    values=[schedule_value(.1,k) for k in range(100)]
    assert values[0]==.2 and values[40]==.1
    assert values[79]==pytest.approx(.0025)
    assert values[80:]==[0.]*20 and sum(v>0 for v in values)==80
    assert math.isclose(sum(values)/100,.081)
    for kwargs in ({'base_alpha':.75,'index':0},{'base_alpha':.1,'index':-1},
                   {'base_alpha':.1,'index':100},{'base_alpha':.1,'index':False},
                   {'base_alpha':.1,'index':0,'cutoff':100}):
        with pytest.raises(ValueError):schedule_value(**kwargs)
    assert schedule_value(.75,0,policy='fixed')==.75


def test_schedule_survives_reorder_filter_and_split_and_detects_content_change():
    questions=[{'id':f'q{i}','question':f'question {i}'} for i in range(100)]
    schedule=freeze_schedule(questions);validate_schedule(schedule)
    assert schedule==freeze_schedule(list(reversed(questions)))
    condition={'alpha':.1,'strength_policy':LINEAR,'schedule_id':schedule['schedule_id']}
    assigned={r['id']:row_strength(condition,r,schedule) for r in questions}
    for subset in (list(reversed(questions[5:10])),questions[:70],questions[70:]):
        assert all(row_strength(condition,r,schedule)==assigned[r['id']] for r in subset)
    changed=copy.deepcopy(schedule);changed['question_indices']['q0']=0
    if changed==schedule:changed['cutoff']=79
    with pytest.raises(ValueError,match='identity'):validate_schedule(changed)
    with pytest.raises(ValueError,match='identity'):row_strength({**condition,'schedule_id':'bad'},questions[0],schedule)
    with pytest.raises(ValueError,match='absent'):row_strength(condition,{'id':'missing'},schedule)


def test_eight_conditions_and_frozen_select_counterparts():
    cfg={'alpha':.1,'layers':[26],'strength_policies':['fixed',LINEAR]}
    conditions=scheme_conditions(cfg)
    assert len(conditions)==8
    for left,right in zip(conditions[:4],conditions[4:]):
        assert all(left[k]==right[k] for k in ('bank','gate','granularity','alpha','layers'))
        assert left['strength_policy']=='fixed' and right['strength_policy']==LINEAR
        assert left['paired_condition_id']==right['id'] and right['paired_condition_id']==left['id']
    assert confirm_with_counterparts(['scheme_07','scheme_02'],conditions,2)==['scheme_07','scheme_02','scheme_03','scheme_06']
    assert confirm_with_counterparts(['scheme_07','scheme_03'],conditions,2)==['scheme_07','scheme_03']
    assert confirm_with_counterparts(['scheme_07'],conditions[6:7],1)==['scheme_07']


def test_prepare_all_strengths_once_reuses_global_vector_and_skips_tail_lookup(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    module=importlib.import_module('run_overlap_ablation')
    from steer_opd import easysteer_static
    schedule=freeze_schedule([{'id':f'q{i}'} for i in range(100)])
    by_index={i:qid for qid,i in schedule['question_indices'].items()}
    rows=[{'id':by_index[i],'key':str(i),'token_ids':[1,2]} for i in (80,40,99,0)]
    masks={r['key']:{'requested_mask':[True,False],'mask_id':'mask','source_binding':'source'} for r in rows}
    vector=torch.tensor([1.,2.]);lookups=[]
    monkeypatch.setattr(module,'candidate_vectors',lambda *a:(lookups.append(a[1]['id']) or ({26:vector},'vector_id')))
    seen=[]
    def spec(row,gates,vectors,**kwargs):
        seen.append((row['key'],kwargs['alpha'],vectors[26]));return kwargs['alpha']
    monkeypatch.setattr(easysteer_static,'build_spec',spec)
    condition={'alpha':.1,'operator':'slerp','strength_policy':LINEAR,'schedule_id':schedule['schedule_id'],
               'gate':'static_overlap','granularity':'global'}
    specs,metadata=module.prepare_engine_requests(rows,masks,None,condition,schedule)
    assert specs==[False,.1,False,.2] and len(lookups)==1
    assert all(item[2] is vector for item in seen)
    assert [r['schedule_index'] for r in metadata]==[80,40,99,0]
    assert [r['direction_lookup_skipped'] for r in metadata]==[True,False,True,False]
    assert not metadata[0]['steering_mask'].any()
    torch.testing.assert_close(vector,torch.tensor([1.,2.]))


def test_heterogeneous_hf_batch_matches_individual_strengths_and_zero_baseline():
    from test_overlap_ablation import tiny,rows
    from steer_opd.static_screen import score_batch
    model=tiny(3);requests=rows();vectors=[{2:torch.arange(16).float()}]*2
    masks=[[True]*len(r['token_ids']) for r in requests]
    actual=score_batch(model,requests,vectors=vectors,gates=masks,alpha=[.2,0.])
    first=score_batch(model,requests[:1],vectors=vectors[:1],gates=masks[:1],alpha=.2)[0]
    baseline=score_batch(model,requests)[1]
    torch.testing.assert_close(actual[0]['logps'],first['logps'],atol=1e-5,rtol=1e-5)
    torch.testing.assert_close(actual[1]['logps'],baseline['logps'],atol=1e-5,rtol=1e-5)


def test_current_overlap_does_not_assign_question_annealing_indices(monkeypatch):
    cfg=read_json(ROOT/'configs/ablation.json')
    from types import SimpleNamespace
    job=make_base_job(cfg,{'student':SimpleNamespace(support_size=151669)},'rows.json')
    assert 'strength_schedule' not in job and len(job['conditions'])==4
    assert all(c['strength_policy']=='fixed' for c in job['conditions'])
    with pytest.raises(ValueError,match='deferred'):
        scheme_conditions({**cfg,'strength_policies':['fixed',LINEAR]})


def test_engine_path_uses_one_whole_call_per_candidate_with_mixed_scales_and_resumes(tmp_path,monkeypatch):
    from test_overlap_ablation import job_fixture,tiny
    from steer_opd import easysteer_static
    from steer_opd.static_screen import score_batch
    module,job=job_fixture(tmp_path,monkeypatch)
    rows=read_json(tmp_path/'rows.json');schedule=freeze_schedule([{'id':r['id']} for r in rows])
    conditions=scheme_conditions({'alpha':.1,'layers':[2],'strength_policies':['fixed',LINEAR]})
    conditions=[c for c in conditions if c['id'] in ('scheme_01','scheme_05')]
    conditions[1]['schedule_id']=schedule['schedule_id']
    job.update(backend='easysteer',conditions=conditions,strength_schedule=schedule)
    prepared=module.prepare(job);model=tiny(3);calls=[];engines=[]
    monkeypatch.setattr(easysteer_static,'require_easysteer',lambda:None)
    monkeypatch.setattr(easysteer_static,'require_easysteer_gpu',lambda:None)
    monkeypatch.setattr(easysteer_static,'create_engine',lambda *a,**kw:engines.append('engine') or model)
    monkeypatch.setattr(easysteer_static,'build_spec',lambda row,gates,vectors,**kw:{'gates':gates,'vectors':vectors,'alpha':kw['alpha']})
    def scores(engine,requests,specs,**kwargs):
        assert len(requests)==len(rows)
        calls.append([s['alpha'] if s else 0 for s in specs])
        if not any(specs):return score_batch(model,requests)
        return score_batch(model,requests,vectors=[s['vectors'] if s else {2:torch.zeros(16)} for s in specs],
            gates=[s['gates'] if s else [False]*len(r['token_ids']) for r,s in zip(requests,specs)],
            alpha=[s['alpha'] if s else 0 for s in specs])
    monkeypatch.setattr(easysteer_static,'fixed_prefix_scores',scores)
    module.run(job,*prepared)
    assert len(engines)==1 and len(calls)==3  # One engine, baseline + two complete candidates.
    assert set(calls[-1])=={0.,.2}
    comparisons=read_json(tmp_path/'result/strength_comparisons.json')
    assert comparisons['scheme_05']['fixed_id']=='scheme_01'
    assert comparisons['scheme_05']['groups']['zero_strength_tail']['overlap']['questions']==1
    assignments=read_json(tmp_path/'result/strength_assignments.json')
    tail_key=next(k for k,v in assignments['scheme_05'].items() if v['effective_alpha']==0)
    tail=torch.load(tmp_path/f'result/scheme_05/{tail_key}.pt',weights_only=True)
    baseline=torch.load(tmp_path/f'result/baseline/{tail_key}.pt',weights_only=True)
    assert tail['direction_lookup_skipped'] and not tail['steering_mask'].any()
    torch.testing.assert_close(tail['logps'],baseline['logps'],atol=1e-5,rtol=1e-5)
    assert all((tmp_path/'result'/c['id']/'comparison/first3.pdf').exists() for c in conditions)
    monkeypatch.setattr(easysteer_static,'create_engine',lambda *a,**kw:pytest.fail('Committed screen loaded engine'))
    module.run(job,*prepared)
    assert len(calls)==3
    damaged=copy.deepcopy(job);damaged['strength_schedule']['cutoff']=0
    with pytest.raises(ValueError):module.run(damaged,*prepared)
