"""Interrupt real commit boundaries on CPU; no GPU/model download is used."""
import copy
import importlib
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from types import ModuleType, SimpleNamespace as NS

import pytest
import torch
import numpy as np

from steer_opd import experiment_runtime as rt
from steer_opd.resumable_core import ROOT, atomic_json, read_json


def test_uniform_42_is_independent_of_request_subset_order_and_seeds_cpu_rngs():
    from steer_opd.compass_campaign import generation_job
    from transformers import AutoTokenizer
    from steer_opd.compass_alignment import ByteLevelCodec
    cfg=read_json(ROOT/'configs/static_dual_overlap.json')
    codec=ByteLevelCodec(AutoTokenizer.from_pretrained(ROOT/'assets/models/Qwen3-4B',local_files_only=True))
    qs=[{'id':str(i),'question':f'{i}+1=?','answer':'1'} for i in range(3)]
    for source in (True,False):
        job=generation_job(cfg,codec,'student',qs,source=source)
        reordered=generation_job(cfg,codec,'student',list(reversed(qs[1:])),source=source)
        assert {r['seed'] for r in job['requests']+reordered['requests']}=={42}
    def values():return random.random(),np.random.rand(),torch.rand(3)
    rt.configure_seed();a=values();rt.configure_seed();b=values()
    assert a[:2]==b[:2];torch.testing.assert_close(a[2],b[2])
    assert rt.child_environment()['PYTHONHASHSEED']=='42'

def test_progress_heartbeat_eta_resume_and_single_writer_lock(tmp_path):
    with pytest.raises(RuntimeError,match='interrupted'):
        with rt.Progress(tmp_path,'teacher',3,initial=1,interval=.02) as p:
            time.sleep(.06)
            state=read_json(tmp_path/'progress.json')
            assert state['started_at'].endswith('+08:00') and state['eta_seconds'] is None
            with pytest.raises(RuntimeError,match='live writer'):
                with rt.Progress(tmp_path,'teacher',3):pass
            p.phase='scoring';p.update(key='q2',tokens=10)
            state=read_json(tmp_path/'progress.json')
            assert state['completed']==state['committed']==2 and state['eta_seconds']>=0 and state['phase']=='scoring'
            raise RuntimeError('interrupted')
    previous=read_json(tmp_path/'progress.json');assert previous['status']=='failed'
    with rt.Progress(tmp_path,'teacher',3,initial=2) as p:p.update(key='q3')
    state=read_json(tmp_path/'progress.json')
    assert state['status']=='complete' and state['resumed']==2
    assert state['first_started_at']==previous['first_started_at']
    assert state['active_elapsed_seconds']>=previous['active_elapsed_seconds']
    events=[json.loads(line) for line in (tmp_path/'events.jsonl').read_text().splitlines()]
    assert any(e['event']=='heartbeat' for e in events)

def test_runtime_guard_preserves_code_bundle_and_rejects_changed_runtime(tmp_path,monkeypatch):
    # Standalone output is allowed outside the checkout in this CPU fixture.
    job={'seed':42};a=rt.initialize_stage(job,tmp_path)
    assert a==rt.initialize_stage(job,tmp_path)
    record=read_json(tmp_path/'reproducibility.json')
    assert Path(record['source_bundle']).is_file() if Path(record['source_bundle']).is_absolute() else (ROOT/record['source_bundle']).is_file()
    original=rt.source_revision
    def changed():
        _,hashes,bodies=original();return 'new_revision',hashes,bodies
    monkeypatch.setattr(rt,'source_revision',changed)
    with pytest.raises(ValueError,match='Code/runtime changed'):rt.initialize_stage(job,tmp_path)

def test_subprocess_tqdm_is_live_and_console_log_is_complete(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    runner=importlib.import_module('run_on_idle_gpus')
    child=subprocess.Popen([sys.executable,'-u','-c',"import sys,time;sys.stdout.write('start\\r');sys.stdout.flush();time.sleep(.15);print('done')"],stdout=subprocess.PIPE)
    writes=[]
    class Console:
        def write(self,text):writes.append((text,child.poll()))
        def flush(self):pass
    with (tmp_path/'stdout.log').open('wb') as log:assert runner.tee_process(child,log,Console())==0
    assert (tmp_path/'stdout.log').read_bytes()==b'start\rdone\n'
    assert any('start' in text and status is None for text,status in writes)

def test_campaign_adopts_its_live_stage_without_launching_duplicate(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    module=importlib.import_module('run_compass_static_panel')
    run=tmp_path/'run';attempt=run/'attempts/reference_teacher_1';attempt.mkdir(parents=True)
    release=attempt/'release';marker=attempt/'process.json'
    code="import time,json,sys\nfrom pathlib import Path\np=Path(sys.argv[1]);r=Path(sys.argv[2])\nwhile not r.exists():time.sleep(.01)\np.write_text(json.dumps({'exit_code':0}))\n"
    child=subprocess.Popen([sys.executable,'-c',code,str(marker),str(release)],cwd=ROOT)
    try:
        start=int(Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split()[19])
        expected=[sys.executable,'scripts/generate_static_sources.py','--job',str(run/'jobs/reference_teacher.json')]
        atomic_json(attempt/'queue.json',{'pid':child.pid,'start_ticks':start,'command':expected})
        monkeypatch.setattr(module,'project_path',lambda p:tmp_path/p)
        monkeypatch.setattr(module,'load_codecs',lambda _:dict.fromkeys(('teacher','reference','student')))
        monkeypatch.setattr(module,'prepare_initial_jobs',lambda *args:[])
        monkeypatch.setattr(module,'generation_job',lambda *args:{})
        monkeypatch.setattr(module,'frozen_json',lambda *args:None)
        monkeypatch.setattr(module.subprocess,'Popen',lambda *args,**kw:pytest.fail('Launched a duplicate stage'))
        events=[]
        class Progress:
            def event(self,event,**kwargs):
                events.append(event)
                if event=='adopting_live_stage':release.touch()
            def update(self,**kwargs):raise RuntimeError('stop_after_adoption')
        with pytest.raises(RuntimeError,match='stop_after_adoption'):
            module._execute({'run_dir':'run','teacher_gpus':2},Progress())
        assert 'adopting_live_stage' in events and read_json(marker)['exit_code']==0
    finally:
        release.touch();child.wait(timeout=5)


@pytest.mark.parametrize('boundary',['data_write','manifest_commit'])
def test_whole_stage_restarts_uncommitted_output_and_reuses_complete_batch(tmp_path,monkeypatch,boundary):
    stage=rt.StageCheckpoint(tmp_path,'fixed')
    records={q:{'value':torch.tensor([i])} for i,q in enumerate(('q1','q2'))}
    with monkeypatch.context() as m:
        if boundary=='data_write':
            original=rt.atomic_torch
            def interrupt(path,value):
                if Path(path).name=='q2.pt':raise KeyboardInterrupt()
                original(path,value)
            m.setattr(rt,'atomic_torch',interrupt)
        else:
            original=rt.atomic_json
            def interrupt(path,value):
                if Path(path)==stage.path:raise KeyboardInterrupt()
                original(path,value)
            m.setattr(rt,'atomic_json',interrupt)
        with pytest.raises(KeyboardInterrupt):stage.save_tensors(records)
    assert (tmp_path/'q1.pt').exists() and stage.load_tensors(records) is None
    stage.save_tensors(records)
    loaded=rt.StageCheckpoint(tmp_path,'fixed').load_tensors(records)
    assert set(loaded)==set(records) and not (tmp_path/'.objects').exists()
    with pytest.raises(ValueError,match='identity'):rt.StageCheckpoint(tmp_path,'changed')
    (tmp_path/'q1.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='missing or changed'):stage.load()


def test_engine_progress_never_commits_results_or_logs_per_request(tmp_path):
    with rt.Progress(tmp_path,'source',2,show_bar=False) as progress:
        before=(tmp_path/'events.jsonl').read_bytes()
        # vLLM also uses this factory for prompt rendering: do not count it twice.
        with progress.engine_tqdm([1,2],desc='Rendering prompts') as bar:
            for _ in bar:pass
        assert progress.done==0
        with progress.engine_tqdm(total=2,desc='Processed prompts') as bar:
            bar.update(1);bar.update(1)
        assert progress.done==2 and progress.committed==0
        assert (tmp_path/'events.jsonl').read_bytes()==before
        assert not (tmp_path/'complete.json').exists()


def test_committed_stage_refuses_overwrite_before_any_tensor_changes(tmp_path):
    stage=rt.StageCheckpoint(tmp_path,'fixed')
    stage.save_tensors({'q':{'value':torch.tensor([1])}})
    before={p.name:p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    with pytest.raises(ValueError,match='cannot be overwritten'):
        stage.save_tensors({'q':{'value':torch.tensor([2])}})
    assert before=={p.name:p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    assert stage.load_tensors(['q'])['q']['value'].item()==1


def test_partial_candidate_is_replayed_as_a_whole_and_completed_baseline_reused(tmp_path,monkeypatch):
    from test_static_overlap import job_fixture,tiny
    from steer_opd import resumable_core,static_plotting
    module,job=job_fixture(tmp_path,monkeypatch);job['conditions']=job['conditions'][:1]
    prepared=module.prepare(job);model=tiny(3)
    monkeypatch.setattr(resumable_core,'require_gpu',lambda:None)
    monkeypatch.setattr(resumable_core,'load_model',lambda _:model)
    monkeypatch.setattr(static_plotting,'plot_comparisons',lambda *args,**kwargs:None)
    original=module.score_native_candidate;seen=[];fail=True
    def score(*args,**kwargs):
        nonlocal fail
        condition=args[5];seen.append((None if condition is None else condition['id'],[r['key'] for r in args[1]]))
        result=original(*args,**kwargs)
        if condition and fail:fail=False;raise KeyboardInterrupt()
        return result
    monkeypatch.setattr(module,'score_native_candidate',score)
    with pytest.raises(KeyboardInterrupt):module.run(job,*prepared)
    marker=tmp_path/'result/baseline/complete.json';before=marker.read_bytes()
    name=job['conditions'][0]['id'];assert not (tmp_path/'result'/name/'complete.json').exists()
    seen.clear();module.run(job,*prepared)
    assert seen==[(name,['r1','r2'])] and marker.read_bytes()==before
    assert read_json(tmp_path/'result'/name/'complete.json')['record_keys']==['r1','r2']


def test_native_easysteer_replays_whole_interrupted_batch_and_never_calls_hf(tmp_path,monkeypatch):
    from test_dual_overlap import codec
    from steer_opd import easysteer_static
    monkeypatch.syspath_prepend(str(ROOT/'scripts'));module=importlib.import_module('score_compass_sources')
    monkeypatch.setattr(module,'project_path',lambda p:tmp_path/p)
    rows=[{'id':key,'key':key,'prompt_token_ids':[1,2],'token_ids':[3,4],'alignment_id':key} for key in ('r1','r2')]
    c=codec();monkeypatch.setattr(module,'prepare_plans',lambda _: (None,None,{'teacher':rows},{'teacher':c,'student':c}))
    monkeypatch.setattr(module,'load_text_scorer',lambda *args:pytest.fail('Production path loaded HF'))
    monkeypatch.setattr(easysteer_static,'create_engine',lambda *args,**kwargs:NS())
    seen=[];fail=True
    def scores(model,batch,specs,**kwargs):
        nonlocal fail
        seen.append([r['key'] for r in batch]);assert 'on_result' not in kwargs
        if fail:fail=False;raise KeyboardInterrupt()
        return [{'ids':torch.arange(16,dtype=torch.int32).repeat(2,1),'logps':(-4.-torch.arange(16)*.1).repeat(2,1),
                 'sampled_logps':torch.full((2,),-5.)} for _ in batch]
    monkeypatch.setattr(easysteer_static,'fixed_prefix_scores',scores)
    job={'optimizer_updates':0,'teacher':'T','teacher_scores':'teacher','probability_backend':'easysteer','seed':42}
    with pytest.raises(KeyboardInterrupt):module.score(job,'teacher')
    assert not (tmp_path/'teacher/r1.pt').exists()
    module.score(job,'teacher');assert seen==[['r1','r2'],['r1','r2']]
    monkeypatch.setattr(easysteer_static,'create_engine',lambda *args,**kwargs:pytest.fail('Complete stage loaded engine'))
    module.score(job,'teacher')


def test_capture_uses_public_capture_once_and_saves_only_after_generate_returns(tmp_path,monkeypatch):
    from steer_opd import easysteer_extraction as ex
    from transformers import AutoConfig
    monkeypatch.setattr(ex,'project_path',lambda p:tmp_path/p)
    monkeypatch.setattr(ex,'encode_content',lambda *args,**kw:([3,4],[True,True],[(0,1),(1,2)]) if kw.get('with_offsets') else ([3,4],[True,True]))
    monkeypatch.setattr(AutoConfig,'from_pretrained',lambda *args,**kw:NS(hidden_size=2))
    vllm=ModuleType('vllm');cap=ModuleType('vllm.capture');cap.SelectSpec=lambda **kw:NS(**kw)
    monkeypatch.setitem(sys.modules,'vllm',vllm);monkeypatch.setitem(sys.modules,'vllm.capture',cap)
    parent=ModuleType('easysteer');hs=ModuleType('easysteer.capture');parent.capture=hs
    monkeypatch.setitem(sys.modules,'easysteer',parent);monkeypatch.setitem(sys.modules,'easysteer.capture',hs)
    finished=False;calls=[]
    class Engine:
        def generate(self,prompts,**kwargs):
            nonlocal finished
            assert not list((tmp_path/'banks/token_activations').rglob('*.pt'))
            assert callable(kwargs['use_tqdm']) and kwargs['steering'] is False
            calls.append(len(prompts));finished=True
            return [NS(prompt_token_ids=p['prompt_token_ids']) for p in prompts]
    monkeypatch.setattr(ex,'create_engine',lambda *args,**kw:Engine())
    def capture(engine,prompts,**kwargs):
        assert kwargs['seed']==42 and kwargs['layers']==[26]
        outputs=engine.generate(prompts,steering=False,use_tqdm=False)
        positions=[s.prompt_positions for s in kwargs['per_prompt_selects']]
        return NS(layer_ids=[26],outputs=outputs,sample_positions=lambda i:positions[i],
                  sample_token_ids=lambda i:[outputs[i].prompt_token_ids[p] for p in positions[i]],
                  sample=lambda i:{26:torch.ones(3,2,dtype=torch.bfloat16)*(i+1)})
    hs.capture=capture
    original=ex.atomic_torch
    def save(path,value):assert finished;original(path,value)
    monkeypatch.setattr(ex,'atomic_torch',save)
    q={'id':'q','student_prompt_ids':[1,2]};records=[{'text':role} for role in ('T','R','S')]
    pooled={};ex.extract_all({'layers':[26],'student':'S0','bank_dir':'banks'},NS(get_vocab=lambda:{i:i for i in range(47)}),[(q,records,{'checkpoint':'S0'})],pooled)
    assert calls==[3] and len(pooled['q']['layers']['26']['pools'])==3
    assert len(list((tmp_path/'banks/token_activations/q').glob('*.pt')))==3


def test_artifact_index_reuses_whole_stage_hashes_without_rereading_tensors(tmp_path,monkeypatch):
    monkeypatch.setattr(rt,'ROOT',tmp_path)
    run=tmp_path/'run';stage=rt.StageCheckpoint(run/'select/one','fixed')
    stage.save_tensors({'q':{'P_overlap':torch.tensor([.5]),'overlap':torch.tensor([.25])}})
    atomic_json(run/'config.json',{'seed':42});atomic_json(tmp_path/'data/questions.json',[]);atomic_json(tmp_path/'data/split.json',{})
    original=rt.file_sha256
    def no_tensor_reread(path):assert Path(path).suffix!='.pt';return original(path)
    monkeypatch.setattr(rt,'file_sha256',no_tensor_reread)
    rt.export_artifacts({'data_dir':'data'},run)
    assert 'select/one/q.pt' in read_json(run/'artifact_index.json')['files']


@pytest.mark.parametrize('kill',['terminate','kill'])
@pytest.mark.parametrize('committed',[False,True])
def test_real_interruption_reuses_only_fully_committed_calls(tmp_path,kill,committed):
    code="""from pathlib import Path
import time,sys,torch
from steer_opd.experiment_runtime import Progress,StageCheckpoint
from steer_opd.resumable_core import atomic_torch
root=Path(sys.argv[1])
with Progress(root,'cpu_call',2,interval=.05) as p:
    s=StageCheckpoint(root/'call','fixed')
    atomic_torch(root/'call/q.pt',{'value':torch.ones(2)})
    if sys.argv[2]=='True':s.commit([root/'call/q.pt'],record_keys=['q'])
    (root/'ready').write_text('ready')
    while True:time.sleep(.05)
"""
    child=subprocess.Popen([sys.executable,'-u','-c',code,str(tmp_path),str(committed)],env={**os.environ,'CUDA_VISIBLE_DEVICES':''},stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    try:
        deadline=time.monotonic()+15
        while not (tmp_path/'ready').exists() and child.poll() is None and time.monotonic()<deadline:time.sleep(.02)
        assert (tmp_path/'ready').exists()
        getattr(child,kill)();child.communicate(timeout=15)
        stage=rt.StageCheckpoint(tmp_path/'call','fixed');saved=stage.load_tensors(['q'])
        assert (saved is not None)==committed
        if saved is None:stage.save_tensors({'q':{'value':torch.ones(2)}})
        assert torch.equal(stage.load_tensors(['q'])['q']['value'],torch.ones(2))
    finally:
        if child.poll() is None:child.kill();child.communicate()
