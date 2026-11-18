"""Regression checks for n-way sampling across worker dispatch and resume."""
import asyncio
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from steer_opd.resumable_core import read_json
from steer_opd.training.rollout_seeds import SEED_FIELD,SeededRolloutManager,trajectory_seeds
from steer_opd.generation_policy import PRESENCE_FIELD


def test_each_question_uses_the_same_four_distinct_seeds_across_steps_and_resume():
    first=trajectory_seeds(1,256,batch_size=64,rollout_n=4)
    second=trajectory_seeds(2,256,batch_size=64,rollout_n=4)
    assert first.reshape(64,4).tolist()==[[42,43,44,45]]*64
    assert np.array_equal(first,second)
    resumed=trajectory_seeds(21,256,batch_size=64,rollout_n=4)
    assert resumed.reshape(64,4).tolist()==[[42,43,44,45]]*64
    assert np.array_equal(resumed,trajectory_seeds(21,256,batch_size=64,rollout_n=4))
    assert set(first)=={42,43,44,45}


def test_eval_keeps_seed42_and_does_not_consume_training_seeds():
    for step in (0,20,100):
        assert trajectory_seeds(step,536,batch_size=64,rollout_n=4,validate=True).tolist()==[42]*536
    assert trajectory_seeds(21,256,batch_size=64,rollout_n=4).tolist()==[42,43,44,45]*64


@pytest.mark.parametrize('step,count',[(0,256),(1,255),(1,257)])
def test_incomplete_or_unidentified_training_batches_are_rejected(step,count):
    with pytest.raises(ValueError,match='complete rollout batch'):
        trajectory_seeds(step,count,batch_size=64,rollout_n=4)


def batch_fixture():
    from steer_opd.training.verl_entry import activate_verl
    activate_verl()
    from verl import DataProto
    from tensordict import TensorDict
    return DataProto(batch=TensorDict({'prompts':torch.ones((8,2),dtype=torch.long)},batch_size=[8]),
        non_tensor_batch={'extra_info':np.array([{'question_id':q} for q in ['a']*4+['b']*4],dtype=object)},
        meta_info={'global_steps':1})


def plan_fixture(tmp_path):
    return {'settings':{'train_batch_size':2,'rollout_n':4,'presence_penalty':2.},'seed':42,'output':str(tmp_path),'binding':'cpu-test'}


def test_seeds_follow_trajectories_through_worker_chunks_and_are_audited(tmp_path):
    batch=batch_fixture()
    from verl import DataProto
    from tensordict import TensorDict
    class Manager:
        def generate_sequences(self,inputs):
            # Four chunks deliberately split same-question groups; run in reverse
            # scheduling order, then restore the manager's public output order.
            chunks=inputs.chunk(4);outputs={}
            for i,chunk in reversed(list(enumerate(chunks))):
                seeds=chunk.non_tensor_batch[SEED_FIELD]
                outputs[i]=DataProto(batch=TensorDict({'responses':torch.tensor([[int(s),1] for s in seeds])},batch_size=[len(seeds)]),
                    non_tensor_batch={SEED_FIELD:seeds.astype(object),
                        PRESENCE_FIELD:chunk.non_tensor_batch[PRESENCE_FIELD].astype(object)})
            return DataProto.concat([outputs[i] for i in sorted(outputs)])
    manager=SeededRolloutManager(Manager(),plan_fixture(tmp_path))
    result=manager.generate_sequences(batch)
    assert result.non_tensor_batch[SEED_FIELD].tolist()==[42,43,44,45]*2
    assert result.non_tensor_batch[PRESENCE_FIELD].tolist()==[2.]*8
    audit=read_json(tmp_path/'metadata/rollout_seeds/train_step_000001.json')
    assert audit['status']=='completed' and audit['unique_responses_per_question']=={'4':2}
    assert audit['question_ids']==['a']*4+['b']*4
    assert audit['presence_penalty']==2.


@pytest.mark.parametrize('wrong_seed,identical',[(True,False),(False,True)])
def test_missing_seed_echo_or_universally_duplicate_responses_fail(tmp_path,wrong_seed,identical):
    batch=batch_fixture()
    class Manager:
        def generate_sequences(self,inputs):
            seeds=inputs.non_tensor_batch[SEED_FIELD]
            return NS(non_tensor_batch={} if wrong_seed else {SEED_FIELD:seeds,PRESENCE_FIELD:inputs.non_tensor_batch[PRESENCE_FIELD]},
                batch={'responses':torch.zeros((8,2),dtype=torch.long) if identical else torch.arange(16).reshape(8,2)})
    manager=SeededRolloutManager(Manager(),plan_fixture(tmp_path))
    with pytest.raises((ValueError,RuntimeError),match='Actual per-request|identical n-way'):
        manager.generate_sequences(batch)


def test_four_async_requests_keep_distinct_seeds_and_temperature_one(monkeypatch):
    batch_fixture()  # Activate the pinned VERL import without any GPU work.
    from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
    from steer_opd.training.verl_agent import TrajectorySeedAgentLoop
    seen=[]
    async def generate(self,params,priority=0,**kwargs):
        seen.append(dict(params));return NS(extra_fields={})
    monkeypatch.setattr(SingleTurnAgentLoop,'run',generate)
    agent=object.__new__(TrajectorySeedAgentLoop);shared={'temperature':1.,'seed':42}
    async def run():return await asyncio.gather(*(agent.run(shared,**{SEED_FIELD:seed,PRESENCE_FIELD:2.}) for seed in range(42,46)))
    outputs=asyncio.run(run())
    assert [row['seed'] for row in seen]==[42,43,44,45]
    assert all(row['temperature']==1. for row in seen) and shared=={'temperature':1.,'seed':42}
    assert [result.extra_fields[SEED_FIELD] for result in outputs]==[42,43,44,45]
    assert all(row['presence_penalty']==2. for row in seen)
    assert [result.extra_fields[PRESENCE_FIELD] for result in outputs]==[2.]*4


@pytest.mark.parametrize('echo',[None,0.,float('nan')])
def test_presence_penalty_must_be_echoed_by_workers(tmp_path,echo):
    batch=batch_fixture()
    class Manager:
        def generate_sequences(self,inputs):
            extras={SEED_FIELD:inputs.non_tensor_batch[SEED_FIELD]}
            if echo is not None:extras[PRESENCE_FIELD]=np.full(8,echo)
            return NS(non_tensor_batch=extras,batch={'responses':torch.arange(16).reshape(8,2)})
    with pytest.raises(ValueError,match='Actual per-request presence penalties'):
        SeededRolloutManager(Manager(),plan_fixture(tmp_path)).generate_sequences(batch)


def test_validation_requests_also_use_penalty_two_and_keep_seed42(tmp_path):
    batch=batch_fixture();batch.meta_info.update(validate=True,global_steps=0)
    class Manager:
        def generate_sequences(self,inputs):
            assert inputs.non_tensor_batch[SEED_FIELD].tolist()==[42]*8
            assert inputs.non_tensor_batch[PRESENCE_FIELD].tolist()==[2.]*8
            return NS(non_tensor_batch={field:inputs.non_tensor_batch[field] for field in (SEED_FIELD,PRESENCE_FIELD)},
                batch={'responses':torch.zeros((8,2),dtype=torch.long)})
    SeededRolloutManager(Manager(),plan_fixture(tmp_path)).generate_sequences(batch)
    audit=read_json(tmp_path/'metadata/rollout_seeds/eval_step_000000.json')
    assert audit['status']=='completed' and audit['presence_penalty']==2.


def test_seed_slots_reject_wrong_question_grouping_before_dispatch(tmp_path):
    batch=batch_fixture()
    batch.non_tensor_batch['extra_info']=np.array([{'question_id':q} for q in ['a','b']*4],dtype=object)
    class Manager:
        def generate_sequences(self,inputs):raise AssertionError('Invalid slots must not reach generation')
    with pytest.raises(ValueError,match='each rollout seed slot'):
        SeededRolloutManager(Manager(),plan_fixture(tmp_path)).generate_sequences(batch)
