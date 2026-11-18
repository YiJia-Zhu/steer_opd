"""Pinned VERL protocol, packed actor loss and global DP gradient normalization."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from steer_opd.resumable_core import ROOT,digest,read_json
from steer_opd.training.config import TrainingLossConfig
from steer_opd.training.verl_adapter import attach_advantages,sampled_policy_loss,verl_loss
from steer_opd.training.verl_plan import VerlSettings,build_verl_plan,verl_overrides,VERL_SOURCE
from test_training_losses import inputs


@pytest.fixture(scope='module')
def upstream():
    if not (ROOT/VERL_SOURCE).is_dir():pytest.skip('Pinned optional VERL source has not been prepared')
    from steer_opd.training.verl_entry import activate_verl
    activate_verl()
    import verl
    return verl


def test_exact_shared_physical_pool_and_batch_divisibility():
    assert VerlSettings().gpu_pool==(0,1)
    with pytest.raises(ValueError,match='divide'):VerlSettings(train_batch_size=3)
    with pytest.raises(ValueError,match='two shared'):VerlSettings(gpu_pool=(0,))
    with pytest.raises(ValueError,match='two shared'):VerlSettings(gpu_pool=(0,1,2,3))
    with pytest.raises(ValueError,match='unique'):VerlSettings(gpu_pool=(0,0))


def test_verl_recipe_uses_one_on_policy_update_and_no_native_id_kl():
    plan=build_verl_plan(read_json(ROOT/'configs/ablation.json'))
    cfg=verl_overrides(plan)
    assert cfg['actor_rollout_ref.actor.ppo_epochs']==1
    assert cfg['actor_rollout_ref.actor.ppo_mini_batch_size']==cfg['data.train_batch_size']
    assert cfg['distillation.enabled'] is False
    assert cfg['algorithm.use_kl_in_reward'] is False
    assert cfg['critic.enable'] is False
    assert cfg['actor_rollout_ref.actor.use_kl_loss'] is False
    assert cfg['actor_rollout_ref.rollout.temperature']==1.
    assert cfg['actor_rollout_ref.rollout.calculate_log_probs'] is True
    assert cfg['data.max_response_length']==16384
    assert cfg['actor_rollout_ref.rollout.max_model_len']>=cfg['data.max_prompt_length']+cfg['data.max_response_length']
    assert plan['settings']['score_max_model_len']>cfg['actor_rollout_ref.rollout.max_model_len']
    assert cfg['actor_rollout_ref.rollout.engine_kwargs.vllm.logprobs_mode']=='raw_logprobs'
    assert cfg['algorithm.rollout_correction']['bypass_mode'] is True
    assert cfg['algorithm.rollout_correction']['rollout_is'] is None
    assert 'student_logprob' not in plan['phase_order']
    assert plan['train_questions']==1024 and plan['data_policy']=='all_train'
    assert 'confirm_questions' not in plan and plan['train_split']=='all'
    assert plan['settings']['gpu_pool']==[0,1]
    assert plan['execution_hold']=='awaiting_user_approval'


def reply_and_batch(kind='traditional_opd'):
    settings,plan,student,teacher=inputs(kind)
    tensors={'responses':torch.tensor([[1,2,3,0,0]]),
             'prompts':torch.tensor([[0,7,8]]),
             'attention_mask':torch.tensor([[0,1,1,1,1,1,1,0]]),
             'response_mask':torch.tensor([[1,1,1,1,0]]),
             'old_log_probs':torch.tensor([[-1.,-3.,-2.,-4.,0.]]),
             'rollout_log_probs':torch.tensor([[-1.,-3.,-2.,-4.,0.]])}
    reply={'alignment':plan,'teacher_logps':teacher}
    if kind=='compass_opd':reply.update(reference_logps=torch.tensor([-1.5,-2.]),student_anchor_logps=torch.tensor([-2.,-2.,-2.,-4.]))
    return settings,reply,SimpleNamespace(batch=tensors,meta_info={})


@pytest.mark.parametrize('kind,expected',[('traditional_opd',[1.5,1.5,1.,0.,0.]),('compass_opd',[.5,1.5,1.,0.,0.])])
def test_scored_observed_tokens_become_frozen_native_student_advantages(kind,expected):
    settings,reply,batch=reply_and_batch(kind)
    attach_advantages(batch,[reply],settings)
    torch.testing.assert_close(batch.batch['advantages'],torch.tensor([expected]))
    assert batch.batch['response_mask'].tolist()==[[1,1,1,0,0]]
    assert batch.meta_info['steer_opd_valid_tokens']==3
    assert not batch.batch['advantages'].requires_grad
    reply['alignment']['source_token_ids'][0]=999
    with pytest.raises(ValueError,match='actual student'):attach_advantages(batch,[reply],settings)


def test_entire_unalignable_batch_cannot_produce_an_optimizer_step():
    settings,reply,batch=reply_and_batch()
    plan=reply['alignment'];plan['alignment_mask']=[False]*4
    plan['units']=[];plan['alignment_id']=digest({k:v for k,v in plan.items() if k!='alignment_id'})
    with pytest.raises(ValueError,match='Entire rollout batch'):attach_advantages(batch,[reply],settings)


def test_uneven_dp_partitions_equal_single_global_token_mean_gradient():
    logps=torch.tensor([[-1.,-2.,-3.],[-4.,-5.,-6.]],requires_grad=True)
    adv=torch.tensor([[.5,1.,-1.],[2.,0.,0.]])
    mask=torch.tensor([[1,1,1],[1,0,0]],dtype=torch.bool)
    direct=sampled_policy_loss(logps,adv,mask,global_valid_tokens=4)
    expected=torch.autograd.grad(direct,logps,retain_graph=True)[0]
    # Each rank sums its microbatch contributions; FSDP then averages the two ranks.
    ranks=[sampled_policy_loss(logps[i:i+1],adv[i:i+1],mask[i:i+1],global_valid_tokens=4,dp_size=2) for i in range(2)]
    actual=torch.autograd.grad(sum(ranks)/2,logps)[0]
    torch.testing.assert_close(actual,expected)


def test_actual_verl_packed_output_shift_and_public_loss_signature(upstream):
    from tensordict import TensorDict
    from verl.utils import tensordict_utils as tu
    from verl.workers.utils.padding import left_right_2_no_padding
    data=TensorDict({'input_ids':torch.tensor([[0,7,1,2,3],[8,9,4,0,0]]),
        'prompts':torch.tensor([[0,7],[8,9]]),'responses':torch.tensor([[1,2,3],[4,0,0]]),
        'attention_mask':torch.tensor([[0,1,1,1,1],[1,1,1,0,0]]),
        'position_ids':torch.tensor([[0,0,1,2,3],[0,1,2,0,0]]),
        'response_mask':torch.tensor([[1,0,1],[1,0,0]]),
        'advantages':torch.tensor([[2.,0.,-1.],[3.,0.,0.]]),
        'rollout_log_probs':torch.tensor([[-1.5,-2.5,-3.5],[-6.5,0.,0.]])},batch_size=[2])
    packed=left_right_2_no_padding(data)
    tu.assign_non_tensor(packed,batch_num_tokens=3,dp_size=2)
    logp=torch.tensor([-1.,-2.,-3.,-4.,-5.,-6.,-7.],requires_grad=True)
    loss,metrics=verl_loss({'log_probs':logp},packed)
    loss.backward()
    expected=torch.zeros(7);expected[0]=-4/3;expected[2]=2/3;expected[5]=-2.
    torch.testing.assert_close(logp.grad,expected)
    assert 'actor/pg_loss' in metrics
    assert metrics['actor/train_rollout_logprob_abs_diff_mean'].aggregate()==pytest.approx(1.)
    assert metrics['actor/train_rollout_logprob_signed_diff_mean'].aggregate()==pytest.approx(1.)
    assert metrics['actor/train_rollout_logprob_abs_diff_max'].aggregate()==pytest.approx(.5)


def test_real_dataproto_preserves_custom_masks_when_sent_to_worker(upstream):
    from verl import DataProto
    settings,reply,simple=reply_and_batch()
    tensors={**simple.batch,'position_ids':torch.arange(8)[None,:],
             'input_ids':torch.tensor([[0,7,8,1,2,3,0,0]])}
    data=DataProto.from_dict(tensors=tensors)
    attach_advantages(data,[reply],settings)
    packet=data.to_tensordict()
    assert packet['response_mask'].tolist()==[[1,1,1,0,0]]
    assert packet['advantages'].tolist()==[[1.5,1.5,1.,0.,0.]]


def test_hydra_config_matches_pinned_fsdp_worker_and_disables_validation(upstream):
    from steer_opd.training.verl_entry import compose_verl
    plan=build_verl_plan(read_json(ROOT/'configs/ablation.json'))
    cfg=compose_verl(plan)
    assert cfg.actor_rollout_ref.actor.strategy=='fsdp'
    assert cfg.trainer.use_v1 is False
    assert cfg.trainer.test_freq==-1 and cfg.trainer.val_before_train is False
    assert cfg.trainer.n_gpus_per_node==2
    assert cfg.actor_rollout_ref.actor.ppo_mini_batch_size==32
    assert cfg.reward.custom_reward_function.name=='compute_score'
    assert cfg.algorithm.rollout_correction.bypass_mode
    assert cfg.algorithm.rollout_correction.rollout_is is None


def test_cpu_parquet_export_includes_every_training_question_without_a_split(tmp_path):
    from steer_opd.training.verl_entry import prepare_data
    import pandas as pd
    plan=build_verl_plan(read_json(ROOT/'configs/ablation.json'))
    plan['output']=str(tmp_path)
    root=prepare_data(plan)
    train=pd.read_parquet(root/'data/train.parquet')
    assert len(train)==1024 and not (root/'data/confirm.parquet').exists()
    train_ids={r['question_id'] for r in train.extra_info}
    assert train_ids=={r['id'] for r in read_json(ROOT/plan['input_data_dir']/'questions_source.json')}
    assert set(train.data_source)=={'steer_opd'}
    assert all(r['ground_truth']=='' for r in train.reward_model)
    prepare_data(plan)  # Same bytes are accepted on CPU resume, without overwriting them.


def test_unaccepted_or_mutated_training_plan_cannot_start_gpu_work():
    from steer_opd.training.verl_entry import validate_receipt
    plan=build_verl_plan(read_json(ROOT/'configs/ablation.json'))
    with pytest.raises(RuntimeError,match='acceptance is pending'):validate_receipt(plan,None)
    plan['settings']['alpha']=.9
    with pytest.raises(ValueError,match='content identity'):validate_receipt(plan,None)


def test_student_request_uses_assigned_seed_without_mutating_shared_sampling_params(upstream,monkeypatch):
    import asyncio
    from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
    from steer_opd.training.verl_agent import TrajectorySeedAgentLoop
    async def run(self,params,priority=0,**kwargs):return SimpleNamespace(extra_fields={},params=params,priority=priority)
    monkeypatch.setattr(SingleTurnAgentLoop,'run',run)
    original={'temperature':1.,'seed':17}
    obj=object.__new__(TrajectorySeedAgentLoop)
    result=asyncio.run(obj.run(original,priority=3,steer_opd_rollout_seed=44,steer_opd_presence_penalty=2.))
    assert result.params['seed']==44 and result.priority==3 and original['seed']==17
    assert result.extra_fields['steer_opd_rollout_seed']==44
    assert result.params['presence_penalty']==2. and 'presence_penalty' not in original
    assert result.extra_fields['steer_opd_presence_penalty']==2.
    with pytest.raises(ValueError,match='Missing driver-assigned'):
        asyncio.run(obj.run(original))
    with pytest.raises(ValueError,match='Missing driver-assigned presence penalty'):
        asyncio.run(obj.run(original,steer_opd_rollout_seed=44))


def test_compass_gpu_receipt_must_cover_both_extra_frozen_scorers(tmp_path):
    from steer_opd.training.verl_entry import validate_receipt
    from steer_opd.resumable_core import atomic_json
    plan=build_verl_plan(read_json(ROOT/'configs/ablation.json'),loss=TrainingLossConfig(loss_type='compass_opd'))
    receipt={'status':'passed','plan_binding':plan['binding'],'loss_type':'compass_opd',
             'roles':['student_fsdp_rollout','teacher_scoring'],
             'student_logprob_source':plan['student_logprob_source'],
             'resource_mode':'shared_two_gpu','phase_handover_test':'passed'}
    path=tmp_path/'receipt.json';atomic_json(path,receipt)
    with pytest.raises(ValueError,match='exact plan'):validate_receipt(plan,path)
    receipt['roles']+=['reference_scoring','student_anchor_scoring'];atomic_json(path,receipt)
    assert validate_receipt(plan,path)==receipt
