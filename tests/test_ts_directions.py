"""T-S raw differences, the four-condition boundary and committed CPU bank reuse."""
import importlib

import pytest
import torch

from steer_opd.compass_banks import make_common_banks
from steer_opd.compass_campaign import scheme_conditions
from steer_opd.direction_bank import bank_vectors
from steer_opd.experiment_runtime import StageCheckpoint
from steer_opd.method_config import DIRECTION_VERSION,Method
from steer_opd.resumable_core import ROOT,atomic_json,atomic_torch,digest


def pairs():
    return {'a':{'usable':True,'layers':{'26':{'pools':[torch.tensor([4.,1.]),torch.tensor([1.,1.])],
                    'norms':[torch.ones(2)*10,torch.ones(3)*10]}}},
            'b':{'usable':True,'layers':{'26':{'pools':[torch.tensor([2.,9.]),torch.tensor([2.,1.])],
                    'norms':[torch.ones(2)*10,torch.ones(3)*10]}}},
            'bad':{'usable':False,'reason':'empty_self_answer'}}


def test_banks_subtract_teacher_self_before_global_average_and_share_filter():
    banks,_=make_common_banks(pairs(),[26],{'checkpoint':'S0'})
    assert set(banks)=={'T-S_global','T-S_per_question'}
    assert all(b['valid_ids']==['a','b'] and b['invalid']=={'bad':'empty_self_answer'} for b in banks.values())
    raw=torch.tensor([1.5,4.])
    torch.testing.assert_close(banks['T-S_global']['layers']['26']['direction'],raw)
    torch.testing.assert_close(bank_vectors(banks['T-S_global'])[0][26],raw/raw.norm())
    torch.testing.assert_close(bank_vectors(banks['T-S_per_question'],'a')[0][26],torch.tensor([1.,0.]))
    assert Method().roles==('teacher','student')


def test_four_condition_ids_and_unsupported_axes_are_rejected():
    cfg={'alpha':.1,'layers':[26]}
    conditions=scheme_conditions(cfg)
    assert [(c['id'],c['direction'],c['gate'],c['granularity']) for c in conditions]==[
        ('scheme_01','T-S','all','global'),('scheme_02','T-S','all','per_question'),
        ('scheme_03','T-S','static_overlap','global'),('scheme_04','T-S','static_overlap','per_question')]
    for number in (5,6,7,8):
        with pytest.raises(ValueError,match='scheme_04'):scheme_conditions({**cfg,'scheme_ids':[f'scheme_{number:02d}']})
    with pytest.raises(ValueError,match='only T-S'):Method(direction='unsupported')
    legacy=pairs();legacy['a']['layers']['26']['pools'].append(torch.ones(2))
    with pytest.raises(ValueError,match='exactly'):make_common_banks(legacy,[26],{})


def test_cpu_bank_entrypoint_consumes_committed_student_pool_and_resumes_without_rebuild(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    module=importlib.import_module('prepare_compass_banks')
    monkeypatch.setattr(module,'project_path',lambda p:tmp_path/p)
    monkeypatch.setattr(module,'initialize_stage',lambda *args:'cpu_runtime')
    monkeypatch.setattr(module,'checkpoint_content_id',lambda *args:'student_weights')
    cfg={'run_dir':'run','data_dir':'data','student':'S0','layers':[26],'minimum_split_questions':1,
         'alpha':.1,'direction':'T-S','direction_version':DIRECTION_VERSION}
    atomic_json(tmp_path/'data/questions.json',[{'id':k} for k in pairs()])
    identity={'checkpoint':'S0','checkpoint_content_id':'student_weights','layer':26,'direction_version':DIRECTION_VERSION}
    path=tmp_path/'run/activations/pooled.pt'
    atomic_torch(path,{'identity':identity,'pooled':pairs()})
    StageCheckpoint(path.parent,digest(identity)).commit([path])
    module.build(cfg)
    assert {p.stem for p in (tmp_path/'run/banks').glob('T-S_*.pt')}=={'T-S_global','T-S_per_question'}
    monkeypatch.setattr(module,'make_common_banks',lambda *a,**kw:pytest.fail('Committed banks were rebuilt'))
    module.build(cfg)
    atomic_torch(path,{'identity':identity,'pooled':{}})
    with pytest.raises(ValueError,match='missing or changed'):module.build(cfg)
