"""Rollout-probability reuse, detached gradients, and global metric reductions."""
from copy import deepcopy

import pytest
import torch
import numpy as np

from steer_opd.training.rollout_logprobs import SOURCE,bind_old_log_probs,difference_sums
from steer_opd.training.verl_adapter import attach_advantages,sampled_policy_loss
from test_verl_training import reply_and_batch
from steer_opd.generation_policy import PRESENCE_FIELD,RAW_PROBABILITY_BASIS


def test_exact_rollout_replaces_a_different_old_policy_and_preserves_gradient_isolation():
    settings,reply,batch=reply_and_batch()
    rollout=torch.tensor([[-2.,-4.,-3.,-1.,float('nan')]],requires_grad=True)
    batch.batch['rollout_log_probs']=rollout
    # This old tensor would produce different advantages if it were still used.
    batch.batch['old_log_probs']=torch.full_like(rollout,-17.)
    attach_advantages(batch,[reply],settings)
    assert batch.meta_info['steer_opd_student_logprob_source']==SOURCE
    assert not batch.batch['old_log_probs'].requires_grad
    assert batch.batch['old_log_probs'].data_ptr()==rollout.data_ptr()
    torch.testing.assert_close(batch.batch['advantages'],torch.tensor([[2.5,2.5,2.,0.,0.]]))
    current=torch.tensor([[-1.1,-2.1,-3.1,float('nan'),float('nan')]],requires_grad=True)
    loss=sampled_policy_loss(current,batch.batch['advantages'],batch.batch['response_mask'],global_valid_tokens=3)
    loss.backward()
    torch.testing.assert_close(current.grad,torch.tensor([[-2.5/3,-2.5/3,-2./3,0.,0.]]))
    assert rollout.grad is None and reply['teacher_logps'].grad is None


@pytest.mark.parametrize('penalty',[0.,2.])
def test_penalty_metadata_does_not_subtract_from_raw_loss_probabilities(penalty):
    settings,reply,batch=reply_and_batch()
    batch.non_tensor_batch={PRESENCE_FIELD:np.array([penalty])}
    attach_advantages(batch,[reply],settings)
    torch.testing.assert_close(batch.batch['advantages'],torch.tensor([[1.5,1.5,1.,0.,0.]]))
    assert batch.meta_info['steer_opd_probability_basis']==RAW_PROBABILITY_BASIS
    assert batch.meta_info['steer_opd_raw_logprobs_are_behavior_logprobs']==(penalty==0.)


def test_processed_sampling_logprobs_cannot_be_relabelled_as_raw():
    _,_,batch=reply_and_batch()
    batch.meta_info['steer_opd_probability_basis']='processed_sampling_logprobs'
    with pytest.raises(ValueError,match='Transformed sampling logprobs'):
        bind_old_log_probs(batch)


@pytest.mark.parametrize('bad', ['missing','shape','integer','nan','infinity','positive','eos_nan'])
def test_missing_or_invalid_generated_logprobs_never_fall_back_to_fsdp(bad):
    _,_,batch=reply_and_batch()
    if bad=='missing':del batch.batch['rollout_log_probs']
    elif bad=='shape':batch.batch['rollout_log_probs']=torch.zeros(1,4)
    elif bad=='integer':batch.batch['rollout_log_probs']=torch.zeros(1,5,dtype=torch.long)
    else:
        index=3 if bad=='eos_nan' else 1
        batch.batch['rollout_log_probs'][0,index]={'nan':float('nan'),'eos_nan':float('nan'),
            'infinity':-float('inf'),'positive':.25}[bad]
    with pytest.raises(ValueError,match='rollout_log_probs|logprobs|vLLM logprob'):
        bind_old_log_probs(batch)


def test_actual_upstream_bypass_and_custom_advantages_use_sampler_probabilities():
    from steer_opd.training.verl_entry import activate_verl
    activate_verl()
    from omegaconf import OmegaConf
    from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode
    settings,reply,batch=reply_and_batch()
    batch.batch['old_log_probs']=torch.full_like(batch.batch['rollout_log_probs'],-17)
    policy=OmegaConf.create({'loss_mode':'vanilla'})
    correction=OmegaConf.create({'bypass_mode':True,'rollout_is':None,'rollout_rs':None})
    apply_bypass_mode(batch,correction,policy)
    attach_advantages(batch,[reply],settings)
    torch.testing.assert_close(batch.batch['advantages'],torch.tensor([[1.5,1.5,1.,0.,0.]]))
    assert policy.loss_mode=='bypass_mode'


def test_train_rollout_metrics_are_token_weighted_across_uneven_microbatches_and_dp():
    from steer_opd.training.verl_entry import activate_verl
    activate_verl()
    from verl.utils.metric import Metric,AggregationType
    rollout=torch.full((4,3),-8.)
    delta=torch.tensor([[1.,99.,99.],[-1.,2.,3.],[4.,-2.,99.],[1.,-4.,99.]])
    mask=torch.tensor([[1,0,0],[1,1,1],[1,1,0],[1,1,0]],dtype=torch.bool)
    current=(rollout+delta).requires_grad_(True)
    current.data[~mask]=float('nan')
    count=int(mask.sum())
    metrics={key:[] for key in ['abs_diff_mean','signed_diff_mean','abs_diff_max']}
    for rows in [(0,1),(2,3)]:
        rank={key:Metric(aggregation=AggregationType.MAX if key=='abs_diff_max' else AggregationType.SUM)
              for key in metrics}
        for index in rows:
            values=difference_sums(current[index:index+1],rollout[index:index+1],mask[index:index+1],
                global_valid_tokens=count,dp_size=2)
            for key,value in values.items():
                assert not value.requires_grad
                rank[key].append(value)
        for key in metrics:metrics[key].append(rank[key])
    assert Metric.aggregate_dp(metrics['abs_diff_mean'])==pytest.approx(float(delta[mask].abs().mean()))
    assert Metric.aggregate_dp(metrics['signed_diff_mean'])==pytest.approx(float(delta[mask].mean()))
    assert Metric.aggregate_dp(metrics['abs_diff_max'])==pytest.approx(float(delta[mask].abs().max()))
    assert Metric.aggregate_dp(metrics['abs_diff_max'])>0


def test_legacy_plan_cannot_silently_switch_probability_sources():
    from steer_opd.resumable_core import ROOT,read_json
    from steer_opd.training.verl_plan import build_verl_plan,verl_overrides
    plan=build_verl_plan(read_json(ROOT/'configs/ablation.json'))
    assert plan['student_logprob_source']==SOURCE
    legacy=deepcopy(plan);legacy.pop('student_logprob_source')
    with pytest.raises(ValueError,match='new training plan'):verl_overrides(legacy)
