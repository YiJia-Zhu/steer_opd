"""Exact Top16 reuse, cross-tokenizer statistics and CPU-only worker execution."""
from types import SimpleNamespace as NS
import math

import numpy as np
import pytest
import torch

from steer_opd.training.overlap_monitor import row_statistics,summarize_statistics,save_top16,load_top16,OverlapMonitor
from steer_opd.compass_alignment import align_response
from test_compass_alignment import ToyCodec


def record(n):
    return {'ids':torch.arange(16,dtype=torch.int32)[None,:].repeat(n,1),
            'logps':torch.full((n,16),-math.log(32),dtype=torch.float32),
            'sampled_logps':torch.full((n,),-math.log(32),dtype=torch.float32)}


def test_existing_span_metric_keeps_eos_unscored_and_raw_mass():
    codec=ToyCodec({0:b'<eos>',**{i:bytes([96+i]) for i in range(1,25)}})
    row={'id':'r','question_id':'q','token_ids':[1,2,0]}
    alignment=align_response(row['token_ids'],codec,{'teacher':codec})
    part=row_statistics(row,alignment,record(3),record(2),{'student':codec,'teacher':codec})
    summary=summarize_statistics([part])
    assert summary['overlap/token_value']==1.
    assert summary['P_overlap/token_value']==pytest.approx(.5)
    assert summary['overlap/coverage']==pytest.approx(2/3)
    assert summary['rollout/H16_cond_native']==pytest.approx(math.log(16))


def test_question_and_token_weights_have_different_denominators():
    parts=[{'question_id':'a','total_tokens':1,'valid_tokens':1,'overlap_sum':1.,'P_overlap_sum':.8,'student_H16_cond_sum':.2},
           {'question_id':'b','total_tokens':3,'valid_tokens':3,'overlap_sum':0.,'P_overlap_sum':0.,'student_H16_cond_sum':1.8}]
    summary=summarize_statistics(parts)
    assert summary['overlap/question_value']==.5 and summary['overlap/token_value']==.25
    assert summary['P_overlap/question_value']==.4 and summary['P_overlap/token_value']==.2
    assert summary['rollout/H16_cond_native']==pytest.approx(.5)


def test_compressed_top16_roundtrip_preserves_tensor_bytes(tmp_path):
    original=[record(3)];path=tmp_path/'top16.pt.gz';save_top16(path,original);restored=load_top16(path)
    for key in original[0]:assert torch.equal(original[0][key],restored[0][key])


def test_top16_agent_requests_head_without_an_extra_generation(monkeypatch):
    import asyncio
    from steer_opd.training.verl_entry import activate_verl
    activate_verl()
    from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
    from steer_opd.training.overlap_agent import Top16AgentLoop
    seen=[]
    async def run(self,params,priority=0,**kwargs):
        seen.append(params)
        return NS(response_ids=[1,2],extra_fields={'response_topk_ids':np.zeros((2,16),dtype=np.int32),
                   'response_topk_log_probs':np.zeros((2,16),dtype=np.float32)})
    monkeypatch.setattr(SingleTurnAgentLoop,'run',run)
    obj=object.__new__(Top16AgentLoop);original={'temperature':1.}
    asyncio.run(obj.run(original,data_source='steer_opd',steer_opd_rollout_seed=43,steer_opd_presence_penalty=2.))
    assert len(seen)==1 and seen[0]['topk_log_probs']==16 and seen[0]['seed']==43
    assert seen[0]['presence_penalty']==2.
    assert original=={'temperature':1.}


@pytest.mark.parametrize('policy,checkpoint_retention',[(None,None),('minimal_v1',None),
    ('minimal_v1','all_models_latest_state_v1')])
def test_parallel_monitor_uses_real_codecs_and_same_response_prefix(tmp_path,monkeypatch,policy,checkpoint_retention):
    from transformers import AutoTokenizer
    from steer_opd.resumable_core import ROOT,read_json
    from steer_opd.compass_alignment import ByteLevelCodec
    cfg=read_json(ROOT/'configs/ablation.json');paths=cfg['tokenizer_paths']
    codecs={role:ByteLevelCodec(AutoTokenizer.from_pretrained(ROOT/path,local_files_only=True)) for role,path in paths.items()}
    ids=codecs['student'].encode('Let x = 2.');alignment=align_response(ids,codecs['student'],{'teacher':codecs['teacher']})
    rows=[{'id':str(i),'question_id':str(i),'token_ids':ids} for i in range(2)]
    s=record(len(ids));t=record(len(alignment['native_response_ids']['teacher']))
    batch=NS(batch={'responses':torch.tensor([ids,ids]),'attention_mask':torch.ones((2,len(ids)),dtype=torch.long),
        'rollout_log_probs':torch.stack([s['sampled_logps']]*2)},non_tensor_batch={
        'response_topk_ids':[s['ids'].numpy()]*2,'response_topk_log_probs':[s['logps'].numpy()]*2})
    replies=[{'alignment':alignment,'teacher_top16':t} for _ in rows]
    plan={'tokenizer_paths':paths,'output':str(tmp_path),'binding':'cpu_test'}
    if policy:plan['retention_policy']=policy
    if checkpoint_retention:plan['checkpoint_retention']=checkpoint_retention
    monkeypatch.setenv('STEER_OPD_METRIC_WORKERS','2');monitor=OverlapMonitor(plan,codecs)
    try:
        metrics=monitor.compute(batch,rows,replies,21)
        if checkpoint_retention:metrics=monitor.compute(batch,rows,replies,22)
    finally:monitor.close()
    assert 0<=metrics['overlap/token_value']<=1 and 0<=metrics['P_overlap/token_value']<=1
    target=tmp_path/('overlap/latest.json' if checkpoint_retention else 'overlap/step_000021/summary.json')
    result=read_json(target)
    assert result['student_policy_step']==(21 if checkpoint_retention else 20) and result['extra_model_forward_calls']==0
    if checkpoint_retention:assert list((tmp_path/'overlap').iterdir())==[target]
    assert (tmp_path/'overlap/step_000021/student_top16.pt.gz').exists()==(policy is None)
    assert ('trajectories' in result)==(policy is None)


def test_resume_monitoring_keeps_training_contract_and_checks_checkpoint_bytes():
    import copy,tempfile
    from pathlib import Path
    from steer_opd.resumable_core import ROOT,read_json,atomic_json,digest
    from steer_opd.training.verl_plan import VerlSettings,build_verl_plan
    from steer_opd.training.resume_handover import bind_resume_source,validate_resume_source
    with tempfile.TemporaryDirectory(dir=ROOT/'.runtime/tmp') as temp:
        base=Path(temp);oldroot=base/'old';checkpoint=oldroot/'checkpoints/global_step_20'
        checkpoint.mkdir(parents=True);(checkpoint/'actor').mkdir()
        (checkpoint/'data.pt').write_bytes(b'fixture')
        for kind in ('model','optim','extra_state'):
            for rank in (0,1):(checkpoint/'actor'/f'{kind}_world_size_2_rank_{rank}.pt').write_bytes(b'fixture')
        (oldroot/'checkpoints/latest_checkpointed_iteration.txt').write_text('20')
        cfg=read_json(ROOT/'configs/ablation.json')
        old=build_verl_plan(cfg,VerlSettings(monitor_evaluation=True));old['assets']={'fixture':'same_assets'}
        old['binding']=digest({k:v for k,v in old.items() if k!='binding'})
        atomic_json(oldroot/'metadata/plan.json',old)
        new=build_verl_plan(cfg,VerlSettings(monitor_evaluation=True,monitor_overlap=True,run_name='new_monitor'))
        new['output']=str(base/'new');new['assets']=old['assets']
        bind_resume_source(new,checkpoint);assert new['resume_source']['step']==20
        validate_resume_source(new)
        changed=copy.deepcopy(new);changed['settings']['alpha']=.5
        with pytest.raises(ValueError,match='hyperparameters'):bind_resume_source(changed,checkpoint)
        changed=copy.deepcopy(new);changed['request_seed_policy']='constant_42'
        with pytest.raises(ValueError,match='request_seed_policy'):bind_resume_source(changed,checkpoint)
        changed=copy.deepcopy(new);changed['student_logprob_source']='fsdp_recomputed_old_policy'
        with pytest.raises(ValueError,match='student_logprob_source'):bind_resume_source(changed,checkpoint)
        changed=copy.deepcopy(new);changed['sampling_policy']['presence_penalty']=0.
        with pytest.raises(ValueError,match='presence penalty'):bind_resume_source(changed,checkpoint)
        atomic_json(oldroot/'metadata/invalid_sampling.json',{'status':'invalid'})
        with pytest.raises(ValueError,match='invalid-sampling'):bind_resume_source(new,checkpoint)
        (oldroot/'metadata/invalid_sampling.json').unlink()
        (checkpoint/'data.pt').write_bytes(b'corrupted')
        with pytest.raises(ValueError,match='checkpoint file changed'):validate_resume_source(new)
