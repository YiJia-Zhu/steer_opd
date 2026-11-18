"""Generation penalties travel with requests and identities, apart from raw scores."""
from copy import deepcopy
from types import SimpleNamespace as NS

import numpy as np
import pytest

from steer_opd.generation_policy import (DEFAULT_PRESENCE_PENALTY,RAW_PROBABILITY_BASIS,
    presence_penalty,validate_training_sampling_policy)
from steer_opd.resumable_core import ROOT,read_json
from steer_opd.training.verl_plan import VerlSettings,build_verl_plan,verl_overrides


def test_default_two_and_explicit_zero_control():
    assert DEFAULT_PRESENCE_PENALTY==presence_penalty()==2.
    assert presence_penalty({'presence_penalty':0.})==0.
    assert presence_penalty({'presence_penalty':np.float64(2.)})==2.
    with pytest.raises(ValueError,match='explicit presence_penalty'):
        presence_penalty({},require_explicit=True)


@pytest.mark.parametrize('value',[True,'2',None,-2.1,2.1,float('nan'),float('inf')])
def test_invalid_penalties_fail_before_generation(value):
    with pytest.raises(ValueError,match='finite number'):
        presence_penalty({'presence_penalty':value})


@pytest.mark.parametrize('role,source',[('teacher',False),('student',True)])
def test_both_sv_source_generation_jobs_include_penalty(role,source):
    from steer_opd.compass_campaign import generation_job
    cfg=read_json(ROOT/'configs/ablation.json')
    tokenizer=NS(apply_chat_template=lambda *a,**kw:[1,2])
    codec=NS(tokenizer=tokenizer,support_size=10,valid_ids=list(range(10)))
    questions=[{'id':'q','question':'Question','answer':'0'}]
    job=generation_job(cfg,codec,role,questions,source=source)
    assert job['presence_penalty']==2.
    assert job['temperature']==job['top_p']==1.
    assert job['logprobs']==(16 if source else None)
    assert generation_job({**cfg,'presence_penalty':0.},codec,role,questions,source=source)['presence_penalty']==0.


def test_training_plan_records_raw_scores_and_penalized_sampling_separately():
    cfg=read_json(ROOT/'configs/ablation.json')
    plan=build_verl_plan(cfg)
    assert plan['schema']=='steer_verl_training_shared_two_gpu_v7'
    policy=validate_training_sampling_policy(plan)
    assert plan['settings']['presence_penalty']==policy['presence_penalty']==2.
    assert policy['probability_basis']==RAW_PROBABILITY_BASIS
    assert policy['raw_logprobs_are_behavior_logprobs'] is False
    assert policy['importance_correction']=='none'
    overrides=verl_overrides(plan)
    assert overrides['actor_rollout_ref.rollout.engine_kwargs.vllm.logprobs_mode']=='raw_logprobs'
    assert overrides['algorithm.rollout_correction']['rollout_is'] is None
    control=build_verl_plan(cfg,VerlSettings(presence_penalty=0.))
    assert control['binding']!=plan['binding']
    assert control['sampling_policy']['raw_logprobs_are_behavior_logprobs'] is True
    assert control['training_loss']==plan['training_loss']


@pytest.mark.parametrize('change',['legacy','mismatched_penalty'])
def test_legacy_or_mismatched_sampling_policy_requires_a_new_plan(change):
    plan=build_verl_plan(read_json(ROOT/'configs/ablation.json'))
    if change=='legacy':plan.pop('sampling_policy')
    else:plan['settings']['presence_penalty']=0.
    with pytest.raises(ValueError,match='new training plan'):
        verl_overrides(plan)


def test_evaluation_binds_effective_penalty_before_loading_a_gpu(monkeypatch,tmp_path):
    from steer_opd import opd_evaluation as module,checkpoint_identity,execution_policy,experiment_runtime,stage_io
    monkeypatch.setattr(execution_policy,'require_execution_allowed',lambda _:None)
    monkeypatch.setattr(checkpoint_identity,'checkpoint_content_id',lambda _:'fixture-weights')
    monkeypatch.setattr(stage_io,'project',lambda value:tmp_path/value)
    monkeypatch.setattr(module,'aime_questions',lambda _:([{'id':'q'}],{'data':'sha'}))
    monkeypatch.setattr(module.importlib.metadata,'version',lambda _:'fixture')
    identities=[];bindings=[]
    original_digest=module.digest
    def digest(identity):
        identities.append(identity)
        return original_digest(identity)
    monkeypatch.setattr(module,'digest',digest)
    class BeforeGPU(Exception):pass
    class Stage:
        def __init__(self,output,binding):
            self.output=output;bindings.append(binding)
        def load(self):raise BeforeGPU
    monkeypatch.setattr(experiment_runtime,'StageCheckpoint',Stage)
    job={'model':'model','output':'step_5_eval/test','dataset_dir':'data','seed':42,
         'generation':{'n':16,'temperature':1.,'top_p':1.,'max_tokens':8}}
    unchanged=deepcopy(job)
    for generation in (job['generation'],{**job['generation'],'presence_penalty':0.}):
        with pytest.raises(BeforeGPU):module.run_evaluation({**job,'generation':generation})
    assert job==unchanged
    assert [identity['job']['generation']['presence_penalty'] for identity in identities]==[2.,0.]
    assert bindings[0]!=bindings[1]
