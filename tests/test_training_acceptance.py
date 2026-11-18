"""Admission identity, bounded acceptance and failed-phase reporting."""
import json
from dataclasses import replace

import pytest

from steer_opd.resumable_core import ROOT,read_json,digest
from steer_opd.training.verl_entry import admitted_pool,validate_acceptance_plan,commit_acceptance
from steer_opd.training.verl_plan import VerlSettings,build_verl_plan
from steer_opd.training.telemetry import phase


def test_uuid_admission_preserves_physical_gpu_order(tmp_path):
    token=tmp_path/'process.json'
    token.write_text(json.dumps({'gpus':[{'index':2,'uuid':'GPU-a'},{'index':3,'uuid':'GPU-b'}]}))
    plan={'settings':{'gpu_pool':[2,3]}}
    assert admitted_pool(plan,'GPU-a,GPU-b',str(token))['gpu_ids']==[2,3]
    assert admitted_pool(plan,'2,3',str(token))['gpu_uuids']==['GPU-a','GPU-b']
    with pytest.raises(RuntimeError,match='visibility'):admitted_pool(plan,'GPU-b,GPU-a',str(token))
    with pytest.raises(RuntimeError,match='different physical'):
        admitted_pool({'settings':{'gpu_pool':[0,1]}},'GPU-a,GPU-b',str(token))


def test_acceptance_cannot_expand_into_unvalidated_full_training():
    settings=VerlSettings(train_batch_size=2,total_steps=2)
    plan=build_verl_plan(read_json(ROOT/'configs/ablation.json'),settings)
    plan['run_purpose']='two_step_gpu_acceptance'
    plan['binding']=digest({k:v for k,v in plan.items() if k!='binding'})
    validate_acceptance_plan(plan)
    plan['settings']['total_steps']=100
    plan['binding']=digest({k:v for k,v in plan.items() if k!='binding'})
    with pytest.raises(ValueError,match='Bounded acceptance'):validate_acceptance_plan(plan)


def test_missing_real_updates_and_checkpoint_cannot_create_passed_receipt(tmp_path):
    (tmp_path/'logs').mkdir();(tmp_path/'logs/training_events.jsonl').write_text('')
    with pytest.raises(RuntimeError,match='evidence is incomplete'):
        commit_acceptance({'output':str(tmp_path)})
    assert not (tmp_path/'metadata/acceptance_receipt.json').exists()


def test_failed_phase_has_end_time_and_is_not_completed(tmp_path):
    with pytest.raises(ValueError):
        with phase(tmp_path,'scoring'):raise ValueError('bad score')
    rows=[json.loads(s) for s in (tmp_path/'training_events.jsonl').read_text().splitlines()]
    assert rows[-1]['status']=='failed' and rows[-1]['seconds']>=0
    assert rows[0]['event']=='phase_start' and rows[-1]['event']=='phase_end'


def test_acceptance_requires_probability_reuse_and_rejects_extra_old_policy_forward(tmp_path):
    from steer_opd.training.rollout_logprobs import SOURCE
    logs=tmp_path/'logs';logs.mkdir()
    folder=tmp_path/'checkpoints/global_step_2/actor';folder.mkdir(parents=True)
    for rank in (0,1):(folder/f'optim_world_size_2_rank_{rank}.pt').write_bytes(b'fixture')
    (tmp_path/'checkpoints/latest_checkpointed_iteration.txt').write_text('2')
    events=[]
    for step in (1,2):
        events.append({'event':'optimizer_update_committed','step':step})
        events.extend({'event':'phase_end','status':'completed','phase':name} for name in
            ['teacher_scoring_phase','student_rollout_logprob_reuse','student_rollout_weight_sync','teacher_sleep_ack'])
    path=logs/'training_events.jsonl'
    path.write_text(''.join(json.dumps(e)+'\n' for e in events))
    plan={'output':str(tmp_path),'binding':'cpu-fixture','run_purpose':'two_step_gpu_acceptance',
          'student_logprob_source':SOURCE}
    receipt=commit_acceptance(plan)
    assert receipt['student_logprob_source']==SOURCE
    events.append({'event':'phase_end','status':'completed','phase':'student_logprob'})
    path.write_text(''.join(json.dumps(e)+'\n' for e in events))
    with pytest.raises(RuntimeError,match='evidence is incomplete'):commit_acceptance(plan)
