"""Stage contracts and actual shared-owner behavior, using CPU-only synthetic artifacts."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from steer_opd import stage_contracts as keys
from steer_opd.artifact_reuse import StageArtifacts,contract
from steer_opd.experiment_runtime import StageCheckpoint
from steer_opd.resumable_core import ROOT,atomic_json,read_json,digest
from steer_opd.stage_io import configure_layout,component_target,project,layout_paths
from steer_opd.stage_campaign import Campaign


@pytest.fixture
def cfg(tmp_path):
    return configure_layout({**read_json(ROOT/'configs/ablation.json'),'output_root':str(tmp_path/'outputs')})


@pytest.fixture
def codecs():
    tokenizer=NS(chat_template='template',apply_chat_template=lambda messages,**kw:[1,2,len(messages[0]['content'])])
    return {r:NS(mapping_sha256=r,support_id=r,support_size=10,valid_ids=list(range(10)),eos={9},special={8,9},tokenizer=tokenizer) for r in ('teacher','student')}


def test_layout_names_and_reserved_eval_stage(cfg):
    assert cfg['run_name']=='ablation' and cfg['param_name']=='n1000_slerp0.1_layer26_seed42_v1'
    assert '/step_1_rollout/ablation/T_Qwen3.5-35B-A3B_S_Qwen3-4B/OpenThoughts3-1.2M/shared/' in cfg['step_dirs']['1']
    assert '/AIME2025/' in cfg['step_dirs']['5']
    for invalid in ('../escape','x/y','a b'):
        with pytest.raises(ValueError):configure_layout({**cfg,'param_name':invalid})
    with pytest.raises(ValueError):configure_layout({**cfg,'run_name':'arbitrary_run'})
    assert '/compar/' in layout_paths({**cfg,'run_name':'compar'})['3']
    assert not project(cfg['step_dirs']['4']).exists() and not project(cfg['step_dirs']['5']).exists()


def all_keys(cfg,codecs):
    q=[{'id':'q1','question':'Question','answer':'0'}];runtime={'torch':'fixed'}
    teacher=keys.generation(cfg,'teacher',q,codecs['teacher'],'T0',runtime)
    student=keys.generation(cfg,'student',q,codecs['student'],'S0',runtime)
    parents={'teacher':digest(teacher),'student':digest(student)}
    capture=keys.capture(cfg,'student','S0',codecs['student'],parents,runtime)
    bank=keys.bank(cfg,'student',digest(capture),['q1'])
    alignment=keys.alignment(cfg,digest(student),codecs)
    native=keys.native(cfg,'T0',codecs['teacher'],{'student':digest(student),'alignment':digest(alignment)},runtime)
    metrics=keys.metrics({'student':digest(student),'alignment':digest(alignment),'native_teacher':digest(native)})
    calibration=keys.calibration(cfg,{'metrics':digest(metrics)},['q1'])
    return {k:digest(v) for k,v in locals().items() if k in ('teacher','student','capture','bank','alignment','native','metrics','calibration')}


@pytest.mark.parametrize('changes,changed',[
    ({'alpha':.2},set()),
    ({'include_teacher_ablation':True,'teacher_layers':[28]},set()),
    ({'layers':[20]},{'capture','bank'}),
    ({'sv_max_context_len':1024},{'capture','bank'}),
    ({'gate_target_rate':.3},{'calibration'}),
    ({'reference_max_tokens':8000},{'teacher','capture','bank'}),
    ({'reference_temperature':.4},{'teacher','capture','bank'}),
    ({'source_max_tokens':4096},{'student','capture','bank','alignment','native','metrics','calibration'}),
    ({'presence_penalty':0.},{'teacher','student','capture','bank','alignment','native','metrics','calibration'}),
    ({'run_name':'param','param_name':'another'},set()),
])
def test_dependency_invalidation_is_local(cfg,codecs,changes,changed):
    before=all_keys(cfg,codecs);after=all_keys({**cfg,**changes},codecs)
    assert {k for k in before if before[k]!=after[k]}==changed


def commit(folder,value):
    folder=project(folder);atomic_json(folder/'result.json',value)
    StageCheckpoint(folder,digest(value)).commit([folder/'result.json'])
    return folder


def test_sha_references_reuse_without_copy_and_reject_tampering(cfg):
    store=StageArtifacts(cfg);science=contract('teacher',{'checkpoint':'T0','prefix':'q'})
    original=commit(component_target(cfg,'teacher'),{'answer':'original'})
    found=store.publish('teacher',science,original)
    other=configure_layout({**cfg,'param_name':'n1000_alpha02'})
    target=component_target(other,'teacher')
    resolved=StageArtifacts(other).bind(target,StageArtifacts(other).find('teacher',science,target),science)
    assert resolved[0]==original
    assert not (target/'result.json').exists()
    assert read_json(target/'metadata/reuse.json')['artifact_id']==found[1]['artifact_id']
    atomic_json(original/'result.json',{'answer':'tampered'})
    with pytest.raises(ValueError,match='changed'):store.find('teacher',science,target)


def test_incomplete_output_is_not_a_reusable_artifact(cfg):
    store=StageArtifacts(cfg);science=contract('teacher',{'p':1});target=component_target(cfg,'teacher')
    atomic_json(project(target)/'result.json',{'unfinished':True})
    assert store.find('teacher',science,target) is None
    with pytest.raises(FileNotFoundError):store.publish('teacher',science,target)


def test_frozen_reference_cannot_change_contract_or_owner(cfg):
    store=StageArtifacts(cfg);science=contract('teacher',{'p':1})
    original=commit(component_target(cfg,'teacher'),{'x':1});found=store.publish('teacher',science,original)
    target=component_target(configure_layout({**cfg,'param_name':'other'}),'teacher')
    store.bind(target,found,science)
    with pytest.raises(ValueError,match='parameters changed'):store.find('teacher',contract('teacher',{'p':2}),target)
    ref=project(target)/'metadata/reuse.json';packet=read_json(ref);packet['owner']=str(project(target).relative_to(ROOT));atomic_json(ref,packet)
    with pytest.raises(ValueError,match='owner changed'):store.find('teacher',science,target)


def test_external_registration_preserves_original_files_and_checks_proofs(cfg,tmp_path):
    store=StageArtifacts(cfg);old=tmp_path/'old';old.mkdir();data=old/'means.pt';data.write_bytes(b'pooled')
    proof=old/'original_manifest.json';atomic_json(proof,{'protocol':'original'})
    before={p.name:p.read_bytes() for p in old.iterdir()};science=contract('capture_student',{'layer':26})
    found=store.publish_external('capture_student',science,old,[data],component_target(cfg,'capture_student'),
                                 proofs=[proof],provenance={'adapter':'test'})
    assert {p.name:p.read_bytes() for p in old.iterdir()}==before
    assert store.validate(project(found[1]['descriptor_path']),science)[0]==old
    atomic_json(proof,{'protocol':'modified'})
    with pytest.raises(ValueError,match='provenance changed'):store.validate(project(found[1]['descriptor_path']))


def synthetic_campaign(cfg,codecs,calls):
    campaign=Campaign.__new__(Campaign)
    campaign.cfg=deepcopy(cfg);campaign.store=StageArtifacts(cfg);campaign.records={}
    campaign.codecs=codecs;campaign.models={'student':'S0','teacher':'T0'};campaign.runtime={'version':'cpu'}
    campaign.progress=NS(event=lambda *a,**kw:None,update=lambda *a,**kw:None)
    def gpu(name,count,command,*,stage_output,**kw):
        calls.append(name)
        if name.startswith('activations'):
            atomic_json(project(stage_output)/'identity.json',{'stage':name})
            pooled=project(stage_output)/'pooled.pt'
            torch.save({'pooled':{'q1':{'usable':True}}},pooled)
            StageCheckpoint(project(stage_output),name).commit([pooled])
        else:commit(stage_output,{'stage':name,'parameter_independent':True})
    def cpu(name,script,file,extra,**kw):
        calls.append(name);job=read_json(file)
        role=extra[1];commit(component_target(job,'bank_'+role),{'bank':role})
    campaign.gpu=gpu;campaign.cpu=cpu
    return campaign


def test_second_parameter_run_skips_gpu_calls_and_uses_actual_shared_roots(cfg,codecs):
    calls=[];questions=[{'id':'q1','question':'Q','answer':'0'}]
    first=synthetic_campaign(cfg,codecs,calls)
    for role in ('teacher','student'):first.generation(role,questions)
    first.directions(questions)
    assert calls==['reference_teacher','source','activations','banks']
    second_cfg=configure_layout({**cfg,'alpha':.3,'param_name':None})
    second=synthetic_campaign(second_cfg,codecs,calls)
    for role in ('teacher','student'):second.generation(role,questions)
    second.directions(questions)
    assert calls==['reference_teacher','source','activations','banks']
    for key in ('teacher','student','capture_student','bank_student'):
        assert first.records[key]['artifact_id']==second.records[key]['artifact_id']
        assert second.cfg['stage_inputs'][key]==first.records[key]['owner']
    third_cfg=configure_layout({**second_cfg,'layers':[20],'param_name':None})
    third=synthetic_campaign(third_cfg,codecs,calls)
    for role in ('teacher','student'):third.generation(role,questions)
    third.directions(questions)
    assert calls[-2:]==['activations','banks'] and len(calls)==6
    assert third.cfg['stage_inputs']['teacher']==first.records['teacher']['owner']


def test_adding_teacher_reader_does_not_repeat_student_gpu_capture(cfg,codecs):
    calls=[];q=[{'id':'q1','question':'Q','answer':'0'}]
    first=synthetic_campaign(cfg,codecs,calls)
    for role in ('teacher','student'):first.generation(role,q)
    first.directions(q)
    second=synthetic_campaign(configure_layout({**cfg,'include_teacher_ablation':True,'param_name':None}),codecs,calls)
    for role in ('teacher','student'):second.generation(role,q)
    second.directions(q)
    assert calls==['reference_teacher','source','activations','banks','activations_teacher','banks_teacher']
    assert second.records['bank_student']['artifact_id']==first.records['bank_student']['artifact_id']


def test_checkpoint_tokenizer_and_runtime_are_generation_dependencies(cfg,codecs):
    q=[{'id':'q','question':'Q'}];runtime={'vllm':'1'}
    original=keys.generation(cfg,'student',q,codecs['student'],'S0',runtime)
    assert original!=keys.generation(cfg,'student',q,codecs['student'],'S1',runtime)
    assert original!=keys.generation(cfg,'student',q,codecs['student'],'S0',{'vllm':'2'})
    changed=deepcopy(codecs['student']);changed.mapping_sha256='different'
    assert original!=keys.generation(cfg,'student',q,changed,'S0',runtime)


def test_reused_bank_uses_candidate_alpha_in_vector_identity():
    from steer_opd.static_screen import candidate_vectors
    bank={'identity':{'step':0},'method':{'direction':'T-S','granularity':'global','alpha':.1,'rho':.03,'seed':42},
          'bank_id':'same','layers':{'26':{'direction':torch.tensor([1.,2.]),'scale':3.}}}
    condition={'direction':'T-S','granularity':'global','layers':[26],'alpha':.3}
    vectors,identity=candidate_vectors(bank,{'id':'q'},condition)
    original,original_id=candidate_vectors(bank,{'id':'q'},{**condition,'alpha':.1})
    assert torch.equal(vectors[26],original[26]) and identity!=original_id


def test_mocked_whole_campaign_reuses_upstreams_and_rebuilds_candidates(cfg,codecs,monkeypatch,tmp_path):
    from steer_opd import stage_campaign as module
    cfg={**cfg,'scheme_ids':['scheme_01'],'confirm_top_n':1}
    questions=[{'id':'q1','question':'Question1','answer':'0'},{'id':'q2','question':'Question2','answer':'0'}]
    monkeypatch.setattr(module,'load_codecs',lambda _:codecs)
    monkeypatch.setattr(module,'checkpoint_content_id',lambda p:Path(p).name)
    monkeypatch.setattr(keys,'runtime_key',lambda _:{'version':'cpu'})
    def data(config,*a):
        folder=project(component_target(config,'data'));store=StageArtifacts(config);science=contract('data',{'frozen':True})
        found=store.find('data',science,folder)
        if found:return store.bind(folder,found,science)
        atomic_json(folder/'questions.json',questions)
        atomic_json(folder/'split.json',{'select_ids':['q1'],'confirm_ids':['q2']})
        return store.publish('data',science,folder,files=[folder/'questions.json',folder/'split.json'])
    monkeypatch.setattr(module,'prepare_data',data)
    def plans(job,*a):
        rows=read_json(project(job['rows']));folder=project(job.get('alignment_source_dir',job['alignment_dir']))
        if not (folder/'manifest.json').exists():atomic_json(folder/'manifest.json',{'aligned':True})
        return rows,{}, {'teacher':rows},codecs
    monkeypatch.setattr(module,'prepare_plans',plans)
    monkeypatch.setattr(module,'AlignedMetricAdapter',lambda *a,**kw:None)
    monkeypatch.setattr(module,'load_rollout_topk',lambda *a:None)
    def metrics(job,row,*a):
        folder=project(job['source_metrics_dir']);atomic_json(folder/(row['key']+'.metrics.json'),{'frozen':'metrics'})
    monkeypatch.setattr(module,'cached_source_metrics',metrics)
    def screen_jobs(config,base,*a):
        folder=project(component_target(config,'calibration'))
        # Production prepare_screen_jobs resolves reused calibration references.
        from steer_opd.stage_io import artifact_dir
        folder=project(artifact_dir(config,'calibration'))
        for name in ('thresholds.json','threshold.json'):
            if not (folder/name).exists():atomic_json(folder/name,{'threshold':.2})
        rows=read_json(project(base['rows']));jobs=[]
        for split,row in zip(('select','confirm'),rows):
            file=project(config['run_dir'])/(split+'_rows.json');atomic_json(file,[row])
            jobs.append({**base,'rows':str(file),'output':str(project(config['run_dir'])/split)})
        return jobs
    monkeypatch.setattr(module,'prepare_screen_jobs',screen_jobs)
    calls=[]
    def gpu(name,count,command,*,stage_output,**kw):
        calls.append(name);folder=project(stage_output)
        file=command[command.index('--config')+1] if '--config' in command else command[command.index('--job')+1]
        job=read_json(file)
        if name in ('source','reference_teacher'):
            files=[]
            for request in job['requests']:
                row={**request,'token_ids':[3,4],'rollout_topk':'topk/'+request['key']+'.pt'}
                path=folder/(request['key']+'.json');atomic_json(path,row);files.append(path)
                path=folder/row['rollout_topk'];path.parent.mkdir(parents=True,exist_ok=True);torch.save({'cpu':True},path);files.append(path)
            StageCheckpoint(folder,name).commit(files)
        elif name=='activations':
            folder.mkdir(parents=True,exist_ok=True);path=folder/'pooled.pt'
            torch.save({'pooled':{q['id']:{'usable':True} for q in questions}},path)
            StageCheckpoint(folder,name).commit([path])
        elif name=='native_teacher':
            folder.mkdir(parents=True,exist_ok=True);files=[]
            for row in read_json(project(job['rows'])):
                path=folder/(row['key']+'.pt');torch.save({'cpu':True},path);files.append(path)
            StageCheckpoint(folder,name).commit(files)
        elif name.startswith('overlap_'):
            commit(folder/'baseline',{'scores':'baseline'})
            summaries={}
            for condition in job['conditions']:
                commit(folder/condition['id'],{'alpha':condition['alpha']})
                values={'questions':1,'question_weighted_value':.3,'question_weighted_delta_pp':1.,
                        'token_weighted_value':.35,'token_weighted_delta_pp':1.}
                summaries[condition['id']]={'condition':condition,'question_weighted_delta_pp':1.,
                    'metrics':{name:values for name in ('overlap','P_overlap')}}
            atomic_json(folder/'summaries.json',summaries)
            atomic_json(folder/'complete.json',{'status':'awaiting_user_review','candidates':len(summaries)})
        else:pytest.fail(name)
    def cpu(name,script,file,extra,**kw):
        calls.append(name);job=read_json(file);commit(component_target(job,'bank_student'),{'frozen':'bank'})
    class Progress:
        done=0;total=12
        def event(self,*a,**kw):pass
        def update(self,**kw):self.done+=1
    module.execute(cfg,Progress(),gpu,cpu)
    initial=list(calls)
    assert initial==['reference_teacher','source','activations','banks','native_teacher','overlap_select_student','overlap_confirm_student']
    second=configure_layout({**cfg,'alpha':.2,'param_name':None})
    module.execute(second,Progress(),gpu,cpu)
    assert calls[len(initial):]==['overlap_select_student','overlap_confirm_student']
    assert read_json(project(second['run_dir'])/'campaign.json')['status']=='awaiting_user_review'
    index=read_json(project(second['run_dir'])/'metadata/artifact_index.json')
    assert index['large_files_copied'] is False and len(index['sources'])==11
    assert (project(second['step_dirs']['6'])/'comparison.csv').exists()
    assert not project(second['step_dirs']['4']).exists() and not project(second['step_dirs']['5']).exists()
