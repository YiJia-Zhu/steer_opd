"""Fail-closed memory handover, sleep ACKs, and official runtime default parity."""
from types import SimpleNamespace

import pytest
import torch

from steer_opd.training.sleep_engine import FrozenEngineResidency
from steer_opd.training.shared_pool import SharedCheckpointManager,SharedPoolHandover
from steer_opd.training.rethinking_defaults import defaults,inherited_overrides
from steer_opd.training.verl_plan import VerlSettings,build_verl_plan,verl_overrides
from steer_opd.training.verl_adapter import token_reward_direct
from steer_opd.resumable_core import ROOT,read_json


class FakeEngine:
    def __init__(self,calls):self.calls=calls;self.asleep=False;self.llm_engine=self
    def sleep(self,level,mode):
        assert level==1 and mode=='wait'
        self.calls.append('teacher_to_cpu');self.asleep=True
    def wake_up(self):self.calls.append('teacher_to_gpu');self.asleep=False
    def is_sleeping(self):return self.asleep


class FakeScorer:
    def __init__(self,calls):self.calls=calls;self.asleep=True
    def ensure_asleep(self):
        if not self.asleep:raise RuntimeError('teacher is awake')
    def begin_shared_phase(self):self.ensure_asleep()
    def end_shared_phase(self):self.ensure_asleep()


def fixtures(calls,manager=None):
    scorer=FakeScorer(calls)
    if manager is None:manager=SimpleNamespace(sleep_replicas=lambda:calls.append('rollout_to_cpu'),
                                             update_weights=lambda:calls.append('rollout_to_gpu'))
    checkpoints=SharedCheckpointManager(manager,scorer)
    actor=SimpleNamespace(to=lambda device,**kw:calls.append(('actor_to_'+device,kw)))
    return scorer,checkpoints,SharedPoolHandover(actor,checkpoints,scorer)


def test_teacher_releases_weights_and_kv_before_next_student_phase():
    calls=[];engine=FakeEngine(calls);residency=FrozenEngineResidency(engine)
    residency.sleep();assert residency.state=='asleep'
    residency.wake();assert residency.state=='awake'
    residency.sleep();assert calls==['teacher_to_cpu','teacher_to_gpu','teacher_to_cpu']


def test_failed_sleep_is_not_reported_as_success():
    engine=FakeEngine([])
    engine.sleep=lambda **kw:None
    residency=FrozenEngineResidency(engine)
    with pytest.raises(RuntimeError,match='acknowledge sleep'):residency.sleep()
    assert residency.state=='transitioning'
    with pytest.raises(RuntimeError,match='confirmed'):residency.wake()


def test_gpu_stage_order_includes_blocking_actor_offload_and_frozen_sleep(tmp_path):
    calls=[];scorer,checkpoints,handover=fixtures(calls)
    handover.event_path=tmp_path/'events.jsonl'
    checkpoints.update_weights()
    checkpoints.sleep_replicas()
    with handover.frozen_scoring():
        calls.append('teacher_scoring');scorer.asleep=False
        scorer.asleep=True;calls.append('teacher_sleep_ack')
    handover.require_student();calls.append('student_update')
    assert calls==['rollout_to_gpu','rollout_to_cpu',
        ('actor_to_cpu',{'model':True,'optimizer':True,'grad':True}),
        'teacher_scoring','teacher_sleep_ack','student_update']
    assert handover.event_path.exists()


def test_no_teacher_wake_while_student_rollout_is_awake():
    calls=[];scorer,checkpoints,handover=fixtures(calls)
    checkpoints.update_weights()
    with pytest.raises(RuntimeError,match='rollout must acknowledge'):
        with handover.frozen_scoring():pytest.fail('teacher must not run')
    assert not any(isinstance(c,tuple) for c in calls)


def test_missing_frozen_sleep_ack_blocks_gradient_update_and_rollout_sync():
    calls=[];scorer,checkpoints,handover=fixtures(calls)
    with pytest.raises(RuntimeError,match='teacher is awake'):
        with handover.frozen_scoring():scorer.asleep=False
    assert handover.phase=='frozen_scoring'
    with pytest.raises(RuntimeError):handover.require_student()
    with pytest.raises(RuntimeError):checkpoints.update_weights()


def test_scoring_error_after_successful_sleep_cannot_apply_an_update():
    calls=[];scorer,checkpoints,handover=fixtures(calls)
    with pytest.raises(ValueError,match='bad score'):
        with handover.frozen_scoring():raise ValueError('bad score')
    assert handover.phase=='student'
    assert 'student_update' not in calls


def test_no_silent_whitening_or_reward_accumulation_in_direct_advantage():
    reward=torch.tensor([[1.,2.,4.]],requires_grad=True);mask=torch.tensor([[1,1,0]])
    adv,ret=token_reward_direct(reward,mask)
    torch.testing.assert_close(adv,torch.tensor([[1.,2.,0.]]));torch.testing.assert_close(ret,adv)
    assert not adv.requires_grad
    r3=reward.detach().unsqueeze(-1).expand(1,3,2)
    a3,_=token_reward_direct(r3,mask)
    assert a3.shape==(1,3,2) and a3[0,2].count_nonzero()==0


def test_official_defaults_and_explicit_length_retention_adaptations():
    plan=build_verl_plan(read_json(ROOT/'configs/ablation.json'))
    cfg=verl_overrides(plan);official=defaults()
    adapted={'data.max_response_length':16384,'actor_rollout_ref.rollout.max_model_len':17408,
             'trainer.log_val_generations':0}
    for key,value in inherited_overrides().items():
        assert cfg[key]==adapted.get(key,value),(key,cfg[key],value)
    assert official['data.max_response_length']==7168
    assert official['actor_rollout_ref.rollout.max_model_len']==8192
    assert official['data.train_batch_size']==64
    assert cfg['data.train_batch_size']==cfg['actor_rollout_ref.actor.ppo_mini_batch_size']==32
    assert cfg['trainer.total_training_steps']==64 and cfg['trainer.total_epochs']==2
    assert cfg['actor_rollout_ref.rollout.n']==official['actor_rollout_ref.rollout.n']==4
    assert cfg['actor_rollout_ref.rollout.gpu_memory_utilization']==.8
    assert official['trainer.save_freq']==20 and cfg['trainer.save_freq']==16
    assert cfg['actor_rollout_ref.actor.fsdp_config.param_offload'] is True
    assert cfg['actor_rollout_ref.actor.fsdp_config.optimizer_offload'] is True
    assert plan['resource_mode']=='shared_two_gpu' and plan['settings']['gpu_pool']==[0,1]
    assert 'student_gpus' not in plan['settings'] and 'teacher_gpus' not in plan['settings']
    assert plan['on_policy']['optimizer_steps_per_rollout_batch']==1
    assert plan['settings']['total_steps']==64


def test_cleanup_only_stops_the_owned_session_with_matching_birth_identity():
    import subprocess,sys
    from steer_opd.training.process_cleanup import cleanup_api,stop_session
    child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],cwd=ROOT,start_new_session=True)
    api=cleanup_api();birth=api.process_ticks(child.pid)
    try:
        stop_session(child.pid,birth+1,child=child,grace=1)
        assert child.poll() is None
        stop_session(child.pid,birth,child=child,grace=1)
        assert child.poll() is not None
    finally:
        if child.poll() is None:child.kill();child.wait()
