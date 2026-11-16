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
    cfg=read_json(ROOT/'configs/ablation.json')
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
    assert rt.child_environment()['VLLM_WORKER_MULTIPROC_METHOD']=='spawn'


def test_engine_initialization_timeout_restores_signal_handler():
    import signal
    previous=signal.getsignal(signal.SIGALRM)
    with pytest.raises(TimeoutError,match='initialization exceeded'):
        with rt.engine_startup_guard(.03):time.sleep(.2)
    assert signal.getsignal(signal.SIGALRM)==previous
    assert signal.getitimer(signal.ITIMER_REAL)==(0.,0.)


def test_mixed_cuda_compiler_and_headers_rejected(tmp_path,monkeypatch):
    path=tmp_path/'cuda';(path/'bin').mkdir(parents=True);(path/'include').mkdir()
    (path/'bin/nvcc').write_text('fake compiler');(path/'include/cuda.h').write_text('#define CUDA_VERSION 13000\n')
    monkeypatch.setenv('STEER_OPD_CUDA_HOME',str(path));monkeypatch.setattr(rt.torch.version,'cuda','13.0')
    rt.cuda_toolkit.cache_clear()
    monkeypatch.setattr(rt.subprocess,'check_output',lambda *a,**kw:'Cuda compilation tools, release 13.5, V13.5.37')
    with pytest.raises(RuntimeError,match='coherent CUDA13.0'):rt.cuda_toolkit(tmp_path)
    monkeypatch.setattr(rt.subprocess,'check_output',lambda *a,**kw:'Cuda compilation tools, release 13.0, V13.0.88')
    assert rt.cuda_toolkit(tmp_path)==path
    rt.cuda_toolkit.cache_clear()

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
    module=importlib.import_module('run_ablation')
    run=tmp_path/'run';attempt=run/'attempts/reference_teacher_1';attempt.mkdir(parents=True)
    release=attempt/'release';marker=attempt/'process.json'
    code="import time,json,sys\nfrom pathlib import Path\np=Path(sys.argv[1]);r=Path(sys.argv[2])\nwhile not r.exists():time.sleep(.01)\np.write_text(json.dumps({'exit_code':0}))\n"
    child=subprocess.Popen([sys.executable,'-c',code,str(marker),str(release)],cwd=ROOT)
    try:
        start=int(Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split()[19])
        expected=[sys.executable,'scripts/generate_static_sources.py','--job',str(run/'jobs/reference_teacher.json')]
        atomic_json(attempt/'queue.json',{'pid':child.pid,'start_ticks':start,'command':expected})
        monkeypatch.setattr(module,'project_path',lambda p:tmp_path/p)
        monkeypatch.setattr(module,'child_environment',lambda *args:os.environ.copy())
        monkeypatch.setattr(module.subprocess,'Popen',lambda *args,**kw:pytest.fail('Launched a duplicate stage'))
        events=[]
        class Progress:
            def event(self,event,**kwargs):
                events.append(event)
                if event=='adopting_live_stage':release.touch()
            def update(self,**kwargs):pass
        gpu,_=module.stage_runners({'run_dir':'run','teacher_gpus':2},Progress())
        gpu('reference_teacher',2,expected[1:])
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
    from test_overlap_ablation import job_fixture,tiny
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


def test_capture_uses_public_capture_once_and_never_writes_hidden_states(tmp_path,monkeypatch):
    layer=26;names=['teacher','student']
    from steer_opd import easysteer_extraction as ex
    from transformers import AutoConfig
    monkeypatch.setattr(ex,'project_path',lambda p:tmp_path/p)
    monkeypatch.setattr(ex,'encode_answer',lambda *args,**kw:{
        'input_ids':[1,2,3,4],'prompt_ids':[1,2],'response_ids':[3,4],'body_mask':[True,True],
        'offsets':[(0,1),(1,2)],'original_response_tokens':10,'retained_response_tokens':2,
        'max_context_len':4,'context_truncated':True,'pooling':'sv_training_all_retained_answer_tokens'})
    monkeypatch.setattr(AutoConfig,'from_pretrained',lambda *args,**kw:NS(hidden_size=2560))
    vllm=ModuleType('vllm');cap=ModuleType('vllm.capture');cap.SelectSpec=lambda **kw:NS(**kw)
    monkeypatch.setitem(sys.modules,'vllm',vllm);monkeypatch.setitem(sys.modules,'vllm.capture',cap)
    parent=ModuleType('easysteer');hs=ModuleType('easysteer.capture');parent.capture=hs
    monkeypatch.setitem(sys.modules,'easysteer',parent);monkeypatch.setitem(sys.modules,'easysteer.capture',hs)
    finished=False;calls=[]
    class Engine:
        llm_engine=NS(vllm_config=NS(scheduler_config=NS(max_num_batched_tokens=16384)))
        def generate(self,prompts,**kwargs):
            nonlocal finished
            assert not list((tmp_path/'banks/token_activations').rglob('*.pt'))
            assert callable(kwargs['use_tqdm']) and kwargs['steering'] is False
            calls.append(len(prompts));finished=True
            return [NS(prompt_token_ids=p['prompt_token_ids']) for p in prompts]
    def engine(*args,**kwargs):
        assert kwargs['batch_invariant'] is True
        assert str(args[0]).endswith('S0')
        return Engine()
    monkeypatch.setattr(ex,'create_engine',engine)
    def capture(engine,prompts,**kwargs):
        assert kwargs['seed']==42 and kwargs['layers']==[layer]
        assert kwargs['device_budget_bytes']>=329818740  # The real failed batch exceeded 256 MiB.
        outputs=engine.generate(prompts,steering=False,use_tqdm=False)
        positions=[s.prompt_positions for s in kwargs['per_prompt_selects']]
        return NS(layer_ids=[layer],outputs=outputs,sample_positions=lambda i:positions[i],
                  sample_token_ids=lambda i:[outputs[i].prompt_token_ids[p] for p in positions[i]],
                  sample=lambda i:{layer:torch.ones(3,2560,dtype=torch.bfloat16)*(i+1)})
    hs.capture=capture
    monkeypatch.setattr(torch,'save',lambda *a,**kw:pytest.fail('Raw hidden states must stay in memory'))
    q={'id':'q','student_prompt_text':'prompt'};records=[{'text':role} for role in names]
    pooled={};ex.extract_all({'layers':[layer],'student':'S0','bank_dir':'banks','sv_max_context_len':4},NS(get_vocab=lambda:{i:i for i in range(47)}),[(q,records,{'checkpoint':'S0'})],pooled)
    assert calls==[len(names)] and len(pooled['q']['layers'][str(layer)]['pools'])==len(names)
    assert finished and not list((tmp_path/'banks').rglob('*.pt'))
    assert 'token_activations' not in pooled['q']


def test_hidden_state_compaction_keeps_pool_and_original_provenance_and_exports(tmp_path,monkeypatch):
    from steer_opd.resumable_core import atomic_torch
    monkeypatch.setattr(rt,'ROOT',tmp_path)
    run=tmp_path/'run';output=run/'activations'
    pool=output/'pooled.pt';atomic_torch(pool,{'means':torch.ones(2,3)})
    raw=[output/'token_activations/q'/f'{role}.pt' for role in ('teacher','student')]
    for path in raw:atomic_torch(path,{'states':torch.ones(10,3)})
    stage=rt.StageCheckpoint(output,'original_binding');stage.commit([pool,*raw],questions=1)
    before=pool.read_bytes();original=read_json(stage.path)
    report=stage.compact_hidden_states()
    assert report['removed_files']==2 and report['freed_bytes']>0
    assert pool.read_bytes()==before and not (output/'token_activations').exists()
    marker=stage.load()
    assert marker['binding']==original['binding'] and marker['hidden_state_storage']=='pooled_only'
    assert read_json(output/'hidden_state_cleanup.json')['original_complete']==original
    rt.export_artifacts({'data_dir':'data'},run)
    index=read_json(run/'artifact_index.json')['files']
    assert 'activations/pooled.pt' in index and not any('token_activations' in name for name in index)
    assert stage.compact_hidden_states()=={'removed_files':0,'freed_bytes':0}
    atomic_torch(pool,{'means':torch.zeros(2,3)})
    with pytest.raises(ValueError,match='Committed stage file'):stage.load()


def test_hidden_state_cleanup_resumes_after_partial_unlink(tmp_path,monkeypatch):
    from steer_opd.resumable_core import atomic_torch
    pool=tmp_path/'pooled.pt';atomic_torch(pool,{'means':torch.ones(2,3)})
    raw=[tmp_path/'token_activations/q'/f'{role}.pt' for role in ('teacher','student')]
    for path in raw:atomic_torch(path,{'states':torch.ones(10,3)})
    stage=rt.StageCheckpoint(tmp_path,'fixed');stage.commit([pool,*raw],questions=1)
    unlink=Path.unlink
    def interrupted(path,*args,**kwargs):
        if path==raw[1]:raise OSError('simulated cleanup interruption')
        return unlink(path,*args,**kwargs)
    monkeypatch.setattr(Path,'unlink',interrupted)
    with pytest.raises(OSError,match='interruption'):stage.compact_hidden_states()
    assert stage.load() is not None and not raw[0].exists() and raw[1].exists()
    committed=stage.path.read_bytes()
    monkeypatch.setattr(Path,'unlink',unlink)
    assert stage.compact_hidden_states()['removed_files']==1
    assert stage.path.read_bytes()==committed and not (tmp_path/'token_activations').exists()


def test_hidden_state_cleanup_refuses_other_artifact_kinds(tmp_path):
    from steer_opd.resumable_core import atomic_torch
    pool=tmp_path/'pooled.pt';atomic_torch(pool,{'means':torch.ones(2,3)})
    unrelated=tmp_path/'token_activations/q/source.pt';atomic_torch(unrelated,{'probabilities':torch.ones(3)})
    stage=rt.StageCheckpoint(tmp_path,'fixed');stage.commit([pool,unrelated],questions=1)
    before=stage.path.read_bytes()
    with pytest.raises(ValueError,match='restricted'):stage.compact_hidden_states()
    assert unrelated.is_file() and stage.path.read_bytes()==before


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


def test_teacher_text_reuse_keeps_original_commit_and_rejects_changed_inputs(tmp_path,monkeypatch):
    from steer_opd import compass_campaign as campaign, checkpoint_identity
    from steer_opd.direction_bank import file_sha256
    from steer_opd.resumable_core import digest
    monkeypatch.setattr(campaign,'project_path',Path)
    monkeypatch.setattr(checkpoint_identity,'checkpoint_content_id',lambda *a,**kw:'weights')
    request={'id':'q','key':'q','seed':42,'prompt_token_ids':[1,2]}
    old={'role':'teacher','model':'T','output':str(tmp_path),'vector':None,'logprobs':None,
         'requests':[request],'model_content_id':'weights','max_tokens':8,'seed':42,
         'metric_version':3,'backend_version':7,'gpu_acceptance_binding':'old_acceptance'}
    atomic_json(tmp_path/'job.json',{'job':old,'hash':digest(old)})
    atomic_json(tmp_path/'reproducibility.json',{'binding':'old_runtime','job_hash':digest(old)})
    atomic_json(tmp_path/'q.json',{**request,'model':'T','model_content_id':'weights',
        'job_hash':digest(old),'steering':False,'token_ids':[3],'text':'answer'})
    stage=rt.StageCheckpoint(tmp_path,digest({'job':old,'runtime':'old_runtime'}))
    stage.commit([tmp_path/'job.json',tmp_path/'q.json'],requests=1)
    before=(tmp_path/'complete.json').read_bytes()
    receipt=tmp_path/'reuse.json'
    atomic_json(receipt,{'schema':'teacher_text_reuse_v1','complete_sha256':file_sha256(stage.path),
        'job_sha256':file_sha256(tmp_path/'job.json'),'generation_contract_sha256':digest(campaign.teacher_text_contract(old)),
        'model_content_id':'weights'})
    current={**old,'metric_version':4,'backend_version':8,'gpu_acceptance_binding':'new_acceptance',
             'reuse_committed_texts':str(receipt)}
    assert campaign.validate_teacher_text_reuse(current)==1
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    generation=importlib.import_module('generate_static_sources')
    monkeypatch.setattr(generation,'project_path',Path)
    def forbid(*a,**kw):pytest.fail('Reusing teacher text must not initialize a new runtime or GPU engine')
    monkeypatch.setattr(generation,'create_engine',forbid)
    monkeypatch.setattr(generation,'initialize_stage',forbid)
    generation.generate(current)
    assert (tmp_path/'complete.json').read_bytes()==before
    with pytest.raises(ValueError,match='settings or requests'):
        campaign.validate_teacher_text_reuse({**current,'max_tokens':9})
    monkeypatch.setattr(checkpoint_identity,'checkpoint_content_id',lambda *a,**kw:'other_weights')
    with pytest.raises(ValueError,match='checkpoint contents'):
        campaign.validate_teacher_text_reuse(current)
    monkeypatch.setattr(checkpoint_identity,'checkpoint_content_id',lambda *a,**kw:'weights')
    row=read_json(tmp_path/'q.json');atomic_json(tmp_path/'q.json',{**row,'text':'tampered'})
    with pytest.raises(ValueError,match='Committed stage file'):
        campaign.validate_teacher_text_reuse(current)


def test_capture_budgets_separate_total_host_rows_from_scheduled_device_workspace():
    from steer_opd.easysteer_extraction import capture_budgets
    # Changing total dataset size must not enlarge the per-forward GPU budget.
    small=capture_budgets(50000,2560,1,16384)
    large=capture_budgets(500000,2560,1,16384)
    assert large['device_budget_bytes']==small['device_budget_bytes']
    assert large['budget_bytes']>=500000*(2560*2+12)
    assert small['device_budget_bytes']>329818740  # Observed failing device requirement.
    # Independent bound for full graph outputs plus two selection/cast workspaces.
    assert small['device_budget_bytes']>=16384*(3*2560*8+12)
    deeper=capture_budgets(50000,2560,2,32768)
    assert deeper['device_budget_bytes']>small['device_budget_bytes']
    assert deeper['budget_bytes']>small['budget_bytes']
    with pytest.raises(ValueError):capture_budgets(0,2560,1,16384)


def test_source_rollout_reuse_preserves_commit_and_checks_probability_identity(tmp_path,monkeypatch):
    from steer_opd import compass_campaign as campaign, checkpoint_identity
    from steer_opd.direction_bank import file_sha256
    from steer_opd.prefix_matching import METRIC_VERSION, RULE_VERSION
    from steer_opd.resumable_core import atomic_torch, digest
    from steer_opd.rollout_topk import SOURCE, commit_topk_response
    monkeypatch.setattr(campaign,'project_path',Path)
    monkeypatch.setattr(checkpoint_identity,'checkpoint_content_id',lambda *a,**kw:'weights')
    monkeypatch.setattr(rt,'source_revision',lambda:('new_code',{},{}))
    request={'id':'q','key':'q_0','seed':42,'prompt_token_ids':[1,2]}
    old={'role':'student','model':'S','output':str(tmp_path),'vector':None,'logprobs':16,
        'save_rollout_top_k':16,'logprobs_mode':'raw_logprobs','support_size':32,
        'requests':[request],'model_content_id':'weights','max_tokens':8,'seed':42,
        'metric_version':METRIC_VERSION,'matching_rule_version':RULE_VERSION,
        'gpu_acceptance_binding':'old_acceptance'}
    atomic_json(tmp_path/'job.json',{'job':old,'hash':digest(old)})
    atomic_json(tmp_path/'reproducibility.json',{'binding':'old_runtime','job_hash':digest(old),'source_revision':'old_code'})
    logps=torch.arange(16,dtype=torch.float32).neg()[None,:]-2
    row=commit_topk_response(tmp_path,{**request,'model':'S','model_content_id':'weights',
        'job_hash':digest(old),'steering':False,'token_ids':[0],'text':'answer',
        'sampled_logps':[float(logps[0,0])],'steering_mask':[False],'token_record_schema':2},
        {'ids':torch.arange(16,dtype=torch.int32)[None,:],'logps':logps,
         'sampled_logps':logps[:,0],'source':SOURCE,'support_size':32,'k':16,'probability_mode':'raw_logprobs'})
    stage=rt.StageCheckpoint(tmp_path,digest({'job':old,'runtime':'old_runtime'}))
    stage.commit([tmp_path/'job.json',tmp_path/'q_0.json',tmp_path/'topk/q_0.pt'],requests=1)
    receipt=tmp_path/'reuse.json'
    def write_receipt():
        atomic_json(receipt,{'schema':'source_rollout_reuse_v1','complete_sha256':file_sha256(stage.path),
            'job_sha256':file_sha256(tmp_path/'job.json'),
            'generation_contract_sha256':digest(campaign.source_rollout_contract(old)),
            'model_content_id':'weights','original_source_revision':'old_code','validated_source_revision':'new_code'})
    write_receipt()
    current={**old,'gpu_acceptance_binding':'new_acceptance','reuse_committed_rollout':str(receipt)}
    before={name:(tmp_path/name).read_bytes() for name in ('job.json','q_0.json','complete.json')}
    assert campaign.validate_source_rollout_reuse(current)==1
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    generation=importlib.import_module('generate_static_sources')
    monkeypatch.setattr(generation,'project_path',Path)
    def forbid(*a,**kw):pytest.fail('Committed student source must be reused without initializing runtime or GPU')
    monkeypatch.setattr(generation,'create_engine',forbid)
    monkeypatch.setattr(generation,'initialize_stage',forbid)
    generation.generate(current)
    assert before=={name:(tmp_path/name).read_bytes() for name in before}
    for changed in ({'max_tokens':9},{'metric_version':METRIC_VERSION-1},{'support_size':33}):
        with pytest.raises(ValueError):campaign.validate_source_rollout_reuse({**current,**changed})
    monkeypatch.setattr(rt,'source_revision',lambda:('another_code',{},{}))
    with pytest.raises(ValueError,match='current code'):campaign.validate_source_rollout_reuse(current)
    monkeypatch.setattr(rt,'source_revision',lambda:('new_code',{},{}))
    monkeypatch.setattr(checkpoint_identity,'checkpoint_content_id',lambda *a,**kw:'other_weights')
    with pytest.raises(ValueError,match='checkpoint contents'):campaign.validate_source_rollout_reuse(current)
    monkeypatch.setattr(checkpoint_identity,'checkpoint_content_id',lambda *a,**kw:'weights')
    # Even a freshly hashed stage manifest cannot hide a broken row/tensor binding.
    payload=torch.load(tmp_path/'topk/q_0.pt',weights_only=True)
    payload['logps']-=.1;payload['sampled_logps']-=.1
    atomic_torch(tmp_path/'topk/q_0.pt',payload)
    with pytest.raises(ValueError,match='Committed stage file'):campaign.validate_source_rollout_reuse(current)
    stage.path.unlink()
    stage.commit([tmp_path/'job.json',tmp_path/'q_0.json',tmp_path/'topk/q_0.pt'],requests=1)
    write_receipt()
    with pytest.raises(ValueError,match='integrity/provenance'):campaign.validate_source_rollout_reuse(current)


def test_capture_keeps_length_and_repeated_answers_but_skips_missing_content(tmp_path,monkeypatch):
    from steer_opd.method_config import DIRECTION_VERSION
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    module=importlib.import_module('capture_student_activations')
    monkeypatch.setattr(module,'project_path',lambda p:tmp_path/p)
    monkeypatch.setattr(module,'checkpoint_content_id',lambda *a,**k:'weights')
    monkeypatch.setattr(module.AutoTokenizer,'from_pretrained',lambda *a,**k:object())
    def reached(*a,**k):raise RuntimeError('reached_runtime')
    def forbid(*a,**k):pytest.fail('Precheck must not load a GPU model')
    monkeypatch.setattr(module,'initialize_stage',reached)
    monkeypatch.setattr(module,'compute_required',forbid)
    monkeypatch.setattr(module,'prompt_text',lambda *a:'prompt')
    monkeypatch.setattr(module,'encode_answer',lambda tok,prompt,text,limit:{
        'response_ids':[1] if text.strip() else [],'context_truncated':len(text)>limit})
    rows=[{'id':'select_ok','question':'q1'},{'id':'confirm_truncated','question':'q2'}]
    atomic_json(tmp_path/'data/questions.json',rows)
    atomic_json(tmp_path/'data/split.json',{'select_ids':['select_ok'],'confirm_ids':['confirm_truncated']})
    for role,folder in [('teacher','references/teacher'),('student','source')]:
        atomic_json(tmp_path/'run'/folder/'complete.json',{'test':role})
        for row in rows:
            key=row['id']+('_0' if role=='student' else '')
            atomic_json(tmp_path/'run'/folder/(key+'.json'),{**row,'model':role,'model_content_id':'weights',
                'finish_reason':'length' if role=='teacher' and row['id']=='confirm_truncated' else 'stop',
                'text':'repeat '*500})
    cfg={'direction':'T-S','direction_version':DIRECTION_VERSION,'run_dir':'run','data_dir':'data',
         'teacher':'teacher','student':'student','tokenizer_paths':{'student':'unused'},'minimum_split_questions':1,
         'sv_max_context_len':2048,'enable_thinking':False}
    with pytest.raises(RuntimeError,match='reached_runtime'):
        module.capture(cfg)
    report=read_json(tmp_path/'run/activations/text_quality.json')
    assert report['ready_for_capture'] and report['paired_usable_questions']==2
    assert report['roles']['teacher']['length_truncated']==1
    assert report['roles']['teacher']['context_truncated']==2
    assert report['finish_reason_correctness_repetition_filter'] is False
    record=tmp_path/'run/references/teacher/confirm_truncated.json'
    atomic_json(record,{**read_json(record),'text':''})
    with pytest.raises(ValueError,match='select=1, confirm=0'):module.capture(cfg)
    report=read_json(tmp_path/'run/activations/text_quality.json')
    assert not report['ready_for_capture']
    assert report['paired_usable_questions']==1
    assert not (tmp_path/'run/activations/pooled.pt').exists()


def test_sv_answer_encoding_matches_upstream_joint_tokenization_and_context_limit():
    from transformers import AutoTokenizer
    from steer_opd.extraction import encode_answer
    from steer_opd.math_data import prompt_text
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'assets/models/Qwen3-4B',local_files_only=True)
    prompt=prompt_text(tokenizer,'What is 2+3?',False)
    text='  <|im_end|><think>reason</think> '+('repeat '*1000)+'  '
    actual=encode_answer(tokenizer,prompt,text,128)
    prefix=tokenizer(prompt,add_special_tokens=False)['input_ids']
    full=tokenizer(prompt+text.strip(),add_special_tokens=False,truncation=True,max_length=128)['input_ids']
    assert actual['input_ids']==full and actual['response_ids']==full[len(prefix):]
    assert actual['body_mask']==[True]*(128-len(prefix))
    assert actual['context_truncated'] and actual['retained_response_tokens']<actual['original_response_tokens']
    assert any(t in tokenizer.all_special_ids for t in actual['response_ids'])
    assert not encode_answer(tokenizer,prompt,'  ',128)['response_ids']
    assert not encode_answer(tokenizer,prompt,text,len(prefix))['response_ids']
