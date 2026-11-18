"""Assign reproducible trajectory seeds before VERL splits a batch among workers."""
from collections import Counter,defaultdict

import numpy as np

from steer_opd.resumable_core import ROOT,atomic_json
from steer_opd.training.telemetry import event
from steer_opd.generation_policy import PRESENCE_FIELD, presence_penalty


SEED_FIELD='steer_opd_rollout_seed'
SEED_POLICY='train_per_question_rollout_slot_base42_eval42_v2'


def trajectory_seeds(step,count,*,batch_size,rollout_n,base_seed=42,validate=False):
    if validate:return np.full(count,base_seed,dtype=np.int64)
    expected=batch_size*rollout_n
    if step<1 or count!=expected:
        raise ValueError('Training seed allocation requires a positive step and the complete rollout batch')
    return base_seed+np.arange(count,dtype=np.int64)%rollout_n


class SeededRolloutManager:
    def __init__(self,manager,plan):self.manager=manager;self.plan=plan

    def __getattr__(self,key):return getattr(self.manager,key)

    def generate_sequences(self,batch):
        validate=bool(batch.meta_info.get('validate',False))
        step=int(batch.meta_info.get('global_steps',-1));s=self.plan['settings']
        seeds=trajectory_seeds(step,len(batch),batch_size=s['train_batch_size'],
            rollout_n=s['rollout_n'],base_seed=self.plan['seed'],validate=validate)
        root=ROOT/self.plan['output'];kind='eval' if validate else 'train'
        path=root/f'metadata/rollout_seeds/{kind}_step_{step:06d}.json'
        ids=[str(info.get('question_id',info.get('index',i)))
             for i,info in enumerate(batch.non_tensor_batch['extra_info'])]
        if not validate:
            allocations=defaultdict(list)
            for qid,seed in zip(ids,seeds):allocations[qid].append(int(seed))
            slots=list(range(self.plan['seed'],self.plan['seed']+s['rollout_n']))
            if len(allocations)!=s['train_batch_size'] or any(sorted(values)!=slots for values in allocations.values()):
                raise ValueError('Each question must receive exactly one request for each rollout seed slot')
        batch.non_tensor_batch[SEED_FIELD]=seeds
        penalty=presence_penalty(s)
        batch.non_tensor_batch[PRESENCE_FIELD]=np.full(len(batch),penalty,dtype=np.float64)
        record={'status':'allocated','step':step,'kind':kind,'policy':SEED_POLICY,
            'plan_binding':self.plan['binding'],'base_seed':self.plan['seed'],
            'seeds':seeds.tolist(),'question_ids':ids,'presence_penalty':penalty}
        atomic_json(path,record)
        result=self.manager.generate_sequences(batch)
        actual=result.non_tensor_batch.get(SEED_FIELD)
        if actual is None or not np.array_equal(np.asarray(actual,dtype=np.int64),seeds):
            raise ValueError('Actual per-request rollout seeds differ from the allocated trajectory order')
        actual_penalties=result.non_tensor_batch.get(PRESENCE_FIELD)
        if actual_penalties is None or not np.array_equal(np.asarray(actual_penalties,dtype=np.float64),batch.non_tensor_batch[PRESENCE_FIELD]):
            raise ValueError('Actual per-request presence penalties differ from the driver-assigned sampling policy')
        if not validate:
            groups=defaultdict(set)
            for qid,tokens in zip(ids,result.batch['responses'].tolist()):groups[qid].add(tuple(tokens))
            counts=dict(Counter(len(values) for values in groups.values()))
            record['unique_responses_per_question']=counts
            record['unique_trajectories']=sum(len(values) for values in groups.values())
            event(root/'logs','student_rollout_diversity',step=step,seeds_min=int(seeds.min()),
                seeds_max=int(seeds.max()),questions=len(groups),trajectories=len(seeds),
                unique_trajectories=record['unique_trajectories'],unique_responses_per_question=counts)
            if s['rollout_n']>1 and all(len(values)==1 for values in groups.values()):
                record['status']='failed_all_questions_have_duplicate_rollouts';atomic_json(path,record)
                raise RuntimeError('All questions still have identical n-way rollouts despite distinct seeds')
        record['status']='completed';atomic_json(path,record)
        return result
