"""Expanded search scope, raw additive semantics and Select-only ranking."""
import importlib
import csv
from copy import deepcopy

import pytest
import torch

from steer_opd.resumable_core import ROOT,atomic_json,read_json
from steer_opd.raw_additive import raw_global_vectors
from steer_opd.static_screen import candidate_vectors
from steer_opd.teacher_screen import teacher_requests


@pytest.fixture
def search(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    return importlib.import_module('run_param_search')


def test_grid_reuses_two_points_without_extra_axes(search):
    cfg=read_json(ROOT/'configs/ablation.json');base,cases=search.search_plan(cfg)
    assert base['run_name']=='param' and base['phase']=='overlap'
    assert len(cases)==24 and len({(c['operator'],c['alpha'],c['teacher_layer']) for c in cases})==24
    assert {c['alpha'] for c in cases}=={.05,.1,.3,.5}
    assert {c['teacher_layer'] for c in cases}=={19,29,39}
    refs=[c for c in cases if c['kind']=='completed_reference']
    assert {(c['operator'],c['alpha'],c['teacher_layer']) for c in refs}=={('slerp',.05,29),('slerp',.1,29)}
    assert all(c['scheme_ids']==['scheme_05'] for c in cases)
    assert all('st_all_global_v2' in c['param_name'] for c in cases)
    broken=deepcopy(cfg);broken['parameter_search']['scheme_ids']=[5,6,7,8]
    with pytest.raises(ValueError,match='all/global'):search.search_plan(broken)
    broken=deepcopy(cfg);broken['parameter_search']['alphas'].append(.1)
    with pytest.raises(ValueError,match='Duplicate'):search.search_plan(broken)
    assert cfg['execution_hold']=='awaiting_user_approval'


def test_summary_uses_select_and_only_selected_scheme(search,tmp_path,monkeypatch):
    cases=[]
    for alpha in (.05,.3):
        run=tmp_path/str(alpha);case={'operator':'slerp','alpha':alpha,'teacher_layer':29,'kind':'completed_reference',
              'param_name':str(alpha),'source_run':str(run),'scheme_ids':['scheme_05']};cases.append(case)
        for split in ('select','confirm'):
            values={}
            for n in (5,6,7,8):
                delta=(2. if alpha==.05 else 1.) if split=='select' else (1. if alpha==.05 else 3.)
                m={'questions':700 if split=='select' else 300,'question_weighted_value':.7,
                   'question_weighted_delta_pp':delta,'token_weighted_delta_pp':delta,'token_weighted_overlap16':.8}
                p={k:v for k,v in m.items() if k!='token_weighted_overlap16'};p['token_weighted_value']=.9
                values[f'scheme_{n:02d}']={'condition':{'gate':'all' if n<7 else 'static_overlap',
                    'granularity':'global' if n%2 else 'per_question'},'metrics':{'overlap':m,'P_overlap':p}}
            atomic_json(run/split/'teacher/summaries.json',values)
            atomic_json(run/split/'teacher/metadata/artifact.json',{'test':True})
    monkeypatch.setattr(search,'completed',lambda *a,**kw:True)
    output=tmp_path/'summary';search.aggregate(output,cases,final=True)
    result=read_json(output/'selected_parameters.json')
    assert result['confirm_used_for_selection'] is False
    assert result['winners']['scheme_05']['alpha']==.05
    with (output/'comparison.csv').open(newline='') as stream:rows=list(csv.DictReader(stream))
    assert len(rows)==8 and {r['scheme'] for r in rows}=={'scheme_05'}
    assert {float(r['value']) for r in rows}=={.7}
    assert {float(r['token_value']) for r in rows if r['metric']=='overlap'}=={.8}
    assert {float(r['token_value']) for r in rows if r['metric']=='P_overlap'}=={.9}


def raw_bank():
    return {'bank_id':'raw','identity':{'step':0},'method':{'direction':'S-T','granularity':'global','seed':42},
        'layers':{'39':{'direction':torch.tensor([3.,4.,0.]),'scale':torch.tensor(999.)}}}


def test_raw_additive_applies_strength_once_without_norm_scaling():
    bank=raw_bank();condition={'direction':'S-T','granularity':'global','operator':'additive',
        'alpha':.3,'layers':[39],'additive_scaling':'raw_mean_difference_v1'}
    vectors,vid=candidate_vectors(bank,{'id':'q'},condition)
    torch.testing.assert_close(vectors[39],torch.tensor([.9,1.2,0.]))
    bank['layers']['39']['scale']=torch.tensor(1.)
    assert candidate_vectors(bank,{'id':'other'},condition)[1]==vid
    assert raw_global_vectors(bank,[39],.5)[1]!=vid
    assert not raw_global_vectors(bank,[39],0)[0][39].any()
    bank['layers']['39']['direction']=torch.zeros(3)
    assert not raw_global_vectors(bank,[39],.5)[0][39].any()


def test_teacher_request_passes_additive_to_engine_without_rescaling():
    row={'id':'q','key':'q','token_ids':[1]};native={'q':{**row,'prompt_token_ids':[7]}}
    masks={'q':{'requested_mask':[True],'mask_id':'m','source_binding':'b'}}
    plans={'q':{'units':[{'valid':True,'ranges':{'student':[0,1],'teacher':[0,1]}}]}}
    condition={'direction':'S-T','granularity':'global','operator':'additive','gate':'all',
        'alpha':.3,'layers':[39],'additive_scaling':'raw_mean_difference_v1'}
    calls=[]
    def spec(r,g,v,**kw):calls.append((g,v,kw));return True
    teacher_requests([row],masks,native,plans,raw_bank(),condition,build_spec=spec)
    assert calls[0][2]=={'operator':'additive','alpha':.3}
    torch.testing.assert_close(calls[0][1][39],torch.tensor([.9,1.2,0.]))
    condition['alpha']=0;calls.clear()
    specs,meta,_,_=teacher_requests([row],masks,native,plans,None,condition,build_spec=spec)
    assert specs==[False] and not calls and meta[0]['direction_lookup_skipped']


def test_launcher_additive_and_final_block_identity(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'));launch=importlib.import_module('launch_ablation')
    from steer_opd.compass_campaign import scheme_conditions
    cfg=launch.build_config(launch.parser().parse_args(['--teacher-ablation','1','--schemes','5',
        '--teacher-layer','39','--operator','additive','--alpha','.5']))
    condition=scheme_conditions(cfg)[0]
    assert condition['layers']==[39] and condition['operator']=='additive'
    assert condition['additive_scaling']=='raw_mean_difference_v1'
    assert 'additive0.5' in cfg['param_name']
    with pytest.raises(ValueError,match='scheme_05'):
        launch.build_config(launch.parser().parse_args(['--operator','additive']))
