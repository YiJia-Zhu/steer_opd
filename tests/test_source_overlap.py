"""Source-only plots, strict scope boundaries and quota-aware artifact checks."""
import copy
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace as NS, ModuleType

import pytest
import torch

from steer_opd.compass_alignment import align_response
from steer_opd.resumable_core import ROOT,atomic_json,atomic_torch,read_json
from steer_opd.static_screen import score_batch
from test_compass_alignment import tiny
from test_dual_overlap import codec


def test_source_job_and_initial_preparation_do_not_schedule_sv(tmp_path,monkeypatch):
    from steer_opd import compass_campaign as campaign
    cfg=read_json(ROOT/'configs/ablation.json')
    cfg.update(stop_after='source_overlap',run_dir=str(tmp_path/'run'))
    monkeypatch.setattr(campaign,'project_path',lambda p:Path(p))
    monkeypatch.setattr(campaign,'prepare_questions',lambda *args:[{'id':'q','question':'Q','answer':'A'}])
    monkeypatch.setattr(campaign,'generation_job',lambda *args,**kw:{'role':'student','source':kw['source']})
    codecs={'student':NS(support_size=151669)}
    campaign.prepare_initial_jobs(cfg,codecs)
    job=read_json(tmp_path/'run/metadata/jobs/alignment_and_scoring.json')
    assert job['stage']=='source_overlap' and job['stop_after']=='source_overlap'
    assert not set(job)&{'banks','conditions','strength_schedule'}
    assert sorted(p.name for p in (tmp_path/'run/metadata/jobs').iterdir())==['alignment_and_scoring.json','source.json']


def test_source_report_keeps_dual_metrics_original_splits_and_resumes_on_cpu(tmp_path,monkeypatch):
    from steer_opd import source_overlap as module
    s,t=codec(),codec(True)
    rows=[{'id':f'q{i}','key':f'r{i}','prompt_token_ids':[1,2],'token_ids':[1,4,0],
           'model':'S0','steering':False,'rollout_topk':f'topk/r{i}.pt'} for i in (1,2)]
    plans={r['key']:align_response(r['token_ids'],s,{'teacher':t}) for r in rows}
    own=[{**r,'token_ids':plans[r['key']]['native_response_ids']['teacher']} for r in rows]
    student_model=tiny(3)
    teacher_model=tiny(4)
    sources=dict(zip([r['key'] for r in rows],score_batch(student_model,rows)))
    native=score_batch(teacher_model,own)
    for row in rows:atomic_torch(tmp_path/'source'/row['rollout_topk'],sources[row['key']])
    for role in ('teacher',):
        atomic_json(tmp_path/role/'complete.json',{'role':role})
        for r,score in zip(rows,native):atomic_torch(tmp_path/role/f"{r['key']}.pt",{**score,'binding':'correct'})
    atomic_json(tmp_path/'source/complete.json',{})
    atomic_json(tmp_path/'alignment/manifest.json',{'rejected':{'bad':'invalid_utf8'}})
    atomic_json(tmp_path/'rows.json',rows)
    atomic_json(tmp_path/'data/split.json',{'select_ids':['q1'],'confirm_ids':['q2']})
    cfg={'run_dir':'run','data_dir':'data','stop_after':'source_overlap'}
    job={'stage':'source_overlap','optimizer_updates':0,'rows':'rows.json','source_dir':'source',
         'alignment_dir':'alignment','teacher_scores':'teacher','source_metrics_dir':str(tmp_path/'alignment'),
         'student':'S0','support_size':47,'require_dual_metrics':True}
    monkeypatch.setattr(module,'project_path',lambda p:tmp_path/p)
    monkeypatch.setattr(module,'initialize_stage',lambda *args:'cpu_runtime')
    monkeypatch.setattr(module,'prepare_plans',lambda *args:(rows,plans,{'teacher':own},{'student':s,'teacher':t}))
    monkeypatch.setattr(module,'load_rollout_topk',lambda output,row,*args:sources[row['key']])
    monkeypatch.setattr(module,'native_score_binding',lambda *args:'correct')
    module.save_source_overlap(cfg,job)
    output=tmp_path/'run/source_overlap'
    summary=read_json(output/'summaries.json')
    assert set(summary['metrics'])=={'overlap','P_overlap'}
    assert summary['metrics']['overlap']['valid_positions']==4
    assert summary['by_split']['select']['P_overlap']['questions']==1
    assert summary['rejected']=={'bad':'invalid_utf8'} and not summary['sv_injection']
    record=torch.load(output/'r1.pt',weights_only=True)
    assert torch.isnan(record['overlap'][-1]) and not record['sv_injection']
    assert all((output/name/'question_means.png').exists() and (output/name/'first3.pdf').exists()
               for name in ('overlap','P_overlap'))
    monkeypatch.setattr(module,'load_rollout_topk',lambda *args:pytest.fail('Committed source report recomputed'))
    module.save_source_overlap(cfg,job)
    atomic_json(tmp_path/'teacher/complete.json',{'changed':True})
    with pytest.raises(ValueError,match='identity'):module.save_source_overlap(cfg,job)


def test_source_only_gpu_validation_never_captures_or_injects(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    module=importlib.import_module('validate_static_gpu')
    from steer_opd import easysteer_static
    s,t=codec(),codec(True)
    s.tokenizer=NS(encode=lambda *args,**kwargs:[1,4])
    rows=[{'id':str(i),'key':str(i),'prompt_token_ids':[1,2],'token_ids':[1,4,0]} for i in range(2)]
    model=tiny(3)
    scores=score_batch(model,rows)
    monkeypatch.setattr(module,'project_path',lambda p:tmp_path/p)
    monkeypatch.setattr(module,'initialize_stage',lambda *args:'cpu_validation')
    monkeypatch.setattr(module,'load_codecs',lambda *args:{'student':s,'teacher':t})
    monkeypatch.setattr(module,'prompt_ids',lambda *args,**kwargs:[1,2])
    def forbid(*args,**kwargs):pytest.fail('Source-only acceptance reached SV capture or injection')
    monkeypatch.setattr(module,'hf_capture',forbid)
    monkeypatch.setattr(easysteer_static,'build_spec',forbid)
    calls=[]
    def fixed(engine,requests,specs,**kwargs):
        assert specs==[False]*len(requests)
        calls.append('unsteered_score')
        return copy.deepcopy(scores[:len(requests)])
    monkeypatch.setattr(easysteer_static,'fixed_prefix_scores',fixed)
    def generate(*args,**kwargs):
        assert kwargs['steering'] is False
        outputs=[]
        for i,(row,score) in enumerate(zip(rows,scores)):
            steps=[]
            for pos,token in enumerate(row['token_ids']):
                entries={int(t):NS(logprob=float(lp),rank=j+1) for j,(t,lp) in enumerate(zip(score['ids'][pos],score['logps'][pos]))}
                if token not in entries:entries[token]=NS(logprob=float(score['sampled_logps'][pos]),rank=17)
                steps.append(entries)
            outputs.append(NS(request_id=str(i),outputs=[NS(token_ids=row['token_ids'],logprobs=steps)]))
        return outputs
    engine=NS(generate=generate,collective_rpc=lambda *a,**k:pytest.fail('Probability RPC must not be used'))
    monkeypatch.setattr(easysteer_static,'create_engine',lambda *args,**kwargs:engine)
    vllm=ModuleType('vllm');vllm.SamplingParams=lambda **kwargs:kwargs
    monkeypatch.setitem(sys.modules,'vllm',vllm)
    module.run({'stage':'static_gpu_acceptance','optimizer_updates':0,'run_dir':'run','student':'S0',
        'gpu_acceptance_binding':'source_only','stop_after':'source_overlap','teacher_gpus':2,
        'gpu_memory_utilization':.75,'layers':[26],'alpha':.1},'student','easysteer')
    report=read_json(tmp_path/'run/acceptance/student/easysteer/report.json')
    assert report['sv_validation'] is False and report['source_replay']['within_parity_limits']
    assert 'capture_layers' not in report and len(calls)==3


def test_storage_guard_uses_destination_availability(tmp_path,monkeypatch):
    from steer_opd import experiment_runtime as runtime
    checked=[]
    monkeypatch.setattr(runtime.shutil,'disk_usage',lambda p:checked.append(p) or NS(free=4*1024**3))
    with pytest.raises(OSError,match='destination quota'):runtime.require_artifact_space(tmp_path/'new',3*1024**3)
    assert checked==[tmp_path]
    assert runtime.require_artifact_space(tmp_path,1024**3)['required_bytes']==1024**3
