"""Teacher-space sign, native mask projection, immutable baseline and CPU replay."""
import copy
import importlib
from types import SimpleNamespace as NS

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from steer_opd.compass_alignment import METRIC, align_response
from steer_opd.compass_banks import make_common_banks
from steer_opd.compass_campaign import make_base_job, role_screen_job, merge_role_summaries
from steer_opd.compass_scoring import AlignedMetricAdapter, AlignedTeacherRecords
from steer_opd.method_config import Method, TEACHER_DIRECTION_VERSION
from steer_opd.resumable_core import ROOT, atomic_json, digest, read_json
from steer_opd.experiment_runtime import StageCheckpoint
from steer_opd.static_masks import freeze_static_mask, selector_for_predictions
from steer_opd.static_screen import score_batch
from steer_opd.teacher_screen import project_teacher_mask, teacher_requests, run_teacher_screen
from test_compass_alignment import ToyCodec


@pytest.fixture
def launcher(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    return importlib.import_module('launch_ablation')


def test_opt_in_adds_four_teacher_space_conditions_and_separate_layer(launcher):
    cfg=launcher.build_config(launcher.parser().parse_args(['--teacher-ablation','1']))
    job=make_base_job(cfg,{'student':NS(support_size=151669)},'rows.json')
    assert cfg['run_dir'].endswith('/n1000_slerp0.1_S26_T29_seed42_v1')
    assert cfg['confirm_top_n']==8 and len(job['conditions'])==8 and len(job['banks'])==4
    assert [c['id'] for c in job['conditions']]==[f'scheme_{i:02d}' for i in range(1,9)]
    assert all(c['direction']=='T-S' and c['layers']==[26] for c in job['conditions'][:4])
    assert all(c['direction']=='S-T' and c['layers']==[29] and c['steer_role']=='teacher'
               and c['direction_version']==TEACHER_DIRECTION_VERSION for c in job['conditions'][4:])
    student=role_screen_job({**job,'output':'select'},'student')
    teacher=role_screen_job({**job,'output':'select'},'teacher')
    assert set(student['banks'])=={'T-S_global','T-S_per_question'}
    assert set(teacher['banks'])=={'S-T_global','S-T_per_question'}
    assert teacher['paired_student_baseline']=='select/student/baseline'
    assert teacher['direction']=='S-T' and student['direction']=='T-S'
    assert teacher['identity']==student['identity']
    with pytest.raises(ValueError,match='teacher layer'):
        launcher.build_config(launcher.parser().parse_args(['--teacher-ablation','1','--teacher-layer','40']))
    assert read_json(ROOT/'configs/ablation.json')['execution_hold']=='awaiting_user_approval'


def test_teacher_bank_means_student_read_by_teacher_minus_teacher_read_by_teacher():
    pooled={'q':{'usable':True,'layers':{'29':{'pools':[torch.tensor([2.,7.,1.]),torch.tensor([5.,3.,1.])],
        'norms':[torch.tensor([10.]),torch.tensor([12.])]}}}}
    banks,scale=make_common_banks(pooled,[29],{'checkpoint':'T0'},direction='S-T')
    assert Method(direction='S-T').roles==('student','teacher')
    torch.testing.assert_close(banks['S-T_global']['layers']['29']['direction'],torch.tensor([-3.,4.,0.]))
    assert banks['S-T_global']['identity']['direction_version']==TEACHER_DIRECTION_VERSION
    assert banks['S-T_global']['method']['direction']=='S-T' and set(scale)=={'29'}


def toy_pair():
    raw={i:bytes([33+i]) for i in range(47)};raw[0]=b'<eos>'
    s=ToyCodec({**raw,1:b'a',2:b'b',3:b'c'})
    t=ToyCodec({**raw,11:b'ab',12:b'c'})
    plan=align_response([1,2,3,0],s,{'teacher':t})
    return s,t,plan


def test_static_mask_is_projected_by_shared_span_and_native_prediction_offset():
    s,t,plan=toy_pair()
    assert project_teacher_mask(plan,[True,True,False,False],2)==[True,False]
    assert project_teacher_mask(plan,[False]*4,2)==[False,False]
    with pytest.raises(ValueError,match='constant'):
        project_teacher_mask(plan,[True,False,False,False],2)
    with pytest.raises(ValueError,match='length'):
        project_teacher_mask(plan,[True],2)
    with pytest.raises(ValueError,match='range'):
        project_teacher_mask(plan,[True,True,False,False],1)
    native={'prompt_token_ids':[1,2,3,4,5],'token_ids':[11,12]}
    assert selector_for_predictions(native,[True,False],mode='fixed_prefix')=={'prompt_positions':[4]}


def test_zero_strength_skips_teacher_direction_and_eos_is_not_claimed_as_mapped_injection():
    _,_,plan=toy_pair();row={'id':'q','key':'q','token_ids':[1,2,3,0]}
    native={**row,'prompt_token_ids':[7,8],'token_ids':[11,12]}
    masks={'q':{'requested_mask':[True,True,False,False],'mask_id':'m','source_binding':'s'}}
    condition={'direction':'S-T','granularity':'global','gate':'static_overlap','alpha':0.,'layers':[29]}
    specs,meta,_,gates=teacher_requests([row],masks,{'q':native},{'q':plan},None,condition,
                                      build_spec=lambda *a,**k:pytest.fail('Zero alpha must not author steering'))
    assert specs==[False] and gates==[[False,False]]
    assert meta[0]['direction_lookup_skipped'] and not meta[0]['steering_mask'].any()
    assert len(meta[0]['native_steering_mask'])==2 and len(meta[0]['steering_mask'])==4


def test_role_summary_merge_keeps_committed_role_files_unchanged(tmp_path,monkeypatch,launcher):
    from steer_opd import compass_campaign
    cfg=launcher.build_config(launcher.parser().parse_args(['--teacher-ablation','1']))
    job={**make_base_job(cfg,{'student':NS(support_size=151669)},'rows'),'output':'select'}
    monkeypatch.setattr(compass_campaign,'project_path',lambda p:tmp_path/p)
    for role in ('student','teacher'):
        part=role_screen_job(job,role)
        atomic_json(tmp_path/part['output']/'complete.json',{'done':True})
        atomic_json(tmp_path/part['output']/'summaries.json',{c['id']:{'condition':c} for c in part['conditions']})
    before=(tmp_path/'select/student/summaries.json').read_bytes()
    assert len(merge_role_summaries(job))==8
    assert (tmp_path/'select/student/summaries.json').read_bytes()==before
    assert len(read_json(tmp_path/'select/summaries.json'))==8


def tiny_teacher():
    torch.manual_seed(31)
    cfg=Qwen3Config(vocab_size=47,hidden_size=24,intermediate_size=48,num_hidden_layers=3,
                   num_attention_heads=3,num_key_value_heads=1,head_dim=8,attention_dropout=0.)
    cfg._attn_implementation='eager'
    return Qwen3ForCausalLM(cfg).eval()


@pytest.mark.parametrize('backend',['hf_batched_static','easysteer'])
def test_teacher_scores_fixed_student_tokens_with_own_width_and_resumes_on_cpu(tmp_path,monkeypatch,backend):
    from steer_opd import teacher_screen,compass_scoring,static_plotting,easysteer_static
    from test_compass_alignment import tiny
    s,t,plan=toy_pair()
    rows=[{'id':'q','key':'q_0','prompt_token_ids':[6,7],'token_ids':[1,2,3,0]}]
    native={'q_0':{'id':'q','key':'q_0','prompt_token_ids':[8,9,10],'token_ids':plan['native_response_ids']['teacher']}}
    student=score_batch(tiny(),rows)[0];teacher=tiny_teacher()
    teacher_base=score_batch(teacher,list(native.values()))[0]
    fixed_student=copy.deepcopy(student)
    adapter=AlignedMetricAdapter({'q_0':plan},s,t,require_dual=True)
    values=AlignedTeacherRecords({'q_0':teacher_base},adapter)
    values.native_rows=native;values.source_scores={'q_0':student}
    metrics=adapter('q_0',student,teacher_base)
    mask=freeze_static_mask(rows[0],metrics['overlap'],1.,
        {'metric':METRIC,'student_checkpoint':'S','teacher_checkpoint':'T','student_tokenizer_sha256':s.mapping_sha256,
         'teacher_tokenizer_sha256':t.mapping_sha256,'support_id':'support','student_score_sha256':'s','teacher_score_sha256':'t'},
        alignment_mask=plan['alignment_mask'],source_metrics={k:metrics[k] for k in ('overlap','P_overlap')})
    pooled={'q':{'usable':True,'layers':{'2':{'pools':[torch.arange(24).float(),torch.zeros(24)],
                                            'norms':[torch.ones(3)*10,torch.ones(3)*10]}}}}
    banks,_=make_common_banks(pooled,[2],{'checkpoint':'T','step':0},direction='S-T')
    conditions=[{'id':'scheme_05','direction':'S-T','steer_role':'teacher','granularity':'global','gate':'all',
                 'layers':[2],'alpha':.1,'operator':'slerp','bank':'S-T_global'},
                {'id':'scheme_07','direction':'S-T','steer_role':'teacher','granularity':'global','gate':'static_overlap',
                 'layers':[2],'alpha':.1,'operator':'slerp','bank':'S-T_global'}]
    job={'output':'out','rows':'rows.json','student':'S','teacher':'T','steer_role':'teacher',
         'support_size':47,'backend':backend,'conditions':conditions,'identity':{'metric':METRIC},
         'rank_metric':'overlap','optimizer_updates':0,'paired_student_baseline':'student_baseline'}
    shared={**student,**metrics,'condition':None,'mask_id':mask['mask_id'],'source_binding':mask['source_binding'],
            'steering_mask':torch.zeros(4,dtype=torch.bool)}
    StageCheckpoint(tmp_path/'student_baseline','student_phase').save_tensors({'q_0':shared})
    atomic_json(tmp_path/'out/manifest.json',{'cpu_teacher_contract':True})
    atomic_json(tmp_path/'rows.json',rows)
    monkeypatch.setattr(teacher_screen,'project_path',lambda p:tmp_path/p)
    loaded=[]
    def load(path,*args):loaded.append(str(path));return teacher
    monkeypatch.setattr(compass_scoring,'load_text_scorer',load)
    def engine(path,support,**kwargs):
        assert kwargs['tensor_parallel_size']==2 and kwargs['batch_invariant'] is False
        return load(path)
    def fixed(model,native,specs,*,progress,**kwargs):
        scores=[]
        for row,spec in zip(native,specs):
            scores.extend(score_batch(model,[row],vectors=[spec[0]] if spec else None,
                gates=[spec[1]] if spec else None,alpha=spec[2] if spec else 0.))
        # Reproduce native tqdm's in-memory forward counter, without a GPU engine.
        progress.done+=len(native)
        return scores
    monkeypatch.setattr(easysteer_static,'require_easysteer_gpu',lambda *args:None)
    monkeypatch.setattr(easysteer_static,'create_engine',engine)
    monkeypatch.setattr(easysteer_static,'fixed_prefix_scores',fixed)
    monkeypatch.setattr(easysteer_static,'build_spec',lambda row,gates,vectors,**kwargs:(vectors,gates,kwargs['alpha']))
    monkeypatch.setattr(static_plotting,'plot_dual_comparisons',lambda *a,**kw:None)
    run_teacher_screen(job,rows,{'q_0':mask},values,banks)
    assert len(loaded)==1 and loaded[0].endswith('/T')
    packet=torch.load(tmp_path/'out/scheme_07/q_0.pt',weights_only=True)
    assert packet['ids'].shape==(2,16) and packet['overlap'].shape==(4,)
    assert packet['native_score_role']=='teacher' and packet['intervention_role']=='teacher'
    assert len(packet['native_steering_mask'])==2 and not packet['steering_mask'][-1]
    assert torch.isnan(packet['overlap'][-1])
    saved=read_json(tmp_path/'out/summaries.json')['scheme_07']
    assert saved['metrics']['overlap']['positions']==3
    assert saved['metrics']['overlap']['injection_prediction_positions']==2
    assert saved['common_baseline']['kind']=='paired_student_arm_baseline'
    assert 'teacher_repeat_drift' in saved and (tmp_path/'out/teacher_repeat_drift.json').exists()
    progress=read_json(tmp_path/'out/progress.json')
    assert progress['completed']==progress['committed']==progress['total']==4
    assert progress['resumed']==1
    for key,value in fixed_student.items():
        if torch.is_tensor(value):torch.testing.assert_close(student[key],value)
    monkeypatch.setattr(compass_scoring,'load_text_scorer',lambda *a:pytest.fail('Complete teacher caches must stay CPU-only'))
    monkeypatch.setattr(easysteer_static,'create_engine',lambda *a,**kw:pytest.fail('Complete teacher caches must not create an engine'))
    run_teacher_screen(job,rows,{'q_0':mask},values,banks)


def test_teacher_capture_uses_teacher_tokenizer_and_student_minus_teacher_order(tmp_path,monkeypatch):
    from steer_opd import easysteer_extraction
    from steer_opd.method_config import DIRECTION_VERSION
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    module=importlib.import_module('capture_student_activations')
    monkeypatch.setattr(module,'project_path',lambda p:tmp_path/p)
    monkeypatch.setattr(module,'checkpoint_content_id',lambda *a,**kw:'weights')
    monkeypatch.setattr(module,'initialize_stage',lambda *a:'cpu_runtime')
    monkeypatch.setattr(module,'compute_required',lambda:None)
    tokenizer=NS(get_vocab=lambda:{'x':1},chat_template='teacher-template')
    tokenizer_paths=[]
    monkeypatch.setattr(module.AutoTokenizer,'from_pretrained',lambda path,**kw:tokenizer_paths.append(str(path)) or tokenizer)
    monkeypatch.setattr(module,'prompt_text',lambda *a:'teacher-prompt')
    monkeypatch.setattr(module,'encode_answer',lambda *a:{'response_ids':[1],'context_truncated':False})
    rows=[{'id':'a','question':'first'},{'id':'b','question':'second'}]
    atomic_json(tmp_path/'data/questions.json',rows)
    atomic_json(tmp_path/'data/split.json',{'select_ids':['a'],'confirm_ids':['b']})
    for role,folder in [('student','source'),('teacher','references/teacher')]:
        atomic_json(tmp_path/'run'/folder/'complete.json',{'role':role})
        for q in rows:
            atomic_json(tmp_path/'run'/folder/(q['id']+('_0' if role=='student' else '')+'.json'),
                {**q,'model':role,'model_content_id':'weights','finish_reason':'stop','text':role+'-answer'})
    def extract(cfg,tok,inputs,pooled):
        assert cfg['reader_role']=='teacher' and cfg['layers']==[29] and tok is tokenizer
        for q,pair,identity in inputs:
            assert [r['text'] for r in pair]==['student-answer','teacher-answer']
            assert q['reader_prompt_text']=='teacher-prompt'
            pooled[q['id']]={'usable':True,'layers':{'29':{'pools':[torch.ones(3)*2,torch.ones(3)],
                         'norms':[torch.ones(1),torch.ones(1)]}},'identity':identity}
        atomic_json(tmp_path/cfg['bank_dir']/'capture_budget.json',{'cpu_fixture':True})
    monkeypatch.setattr(easysteer_extraction,'extract_all',extract)
    cfg={'direction':'T-S','direction_version':DIRECTION_VERSION,'run_dir':'run','data_dir':'data',
         'teacher':'teacher','student':'student','tokenizer_paths':{'teacher':'teacher_tokenizer'},
         'include_teacher_ablation':True,'layers':[26],'teacher_layers':[29],'minimum_split_questions':1,
         'sv_max_context_len':2048,'enable_thinking':False,'extraction_backend':'easysteer'}
    module.capture(cfg,'teacher')
    assert tokenizer_paths[0].endswith('teacher_tokenizer')
    saved=torch.load(tmp_path/'run/activations_teacher/pooled.pt',weights_only=True)
    assert saved['identity']['checkpoint']=='teacher' and saved['identity']['layer']==29
    assert saved['identity']['pool_roles']==['student','teacher']
    assert saved['identity']['direction_version']==TEACHER_DIRECTION_VERSION
    assert read_json(tmp_path/'run/activations_teacher/complete.json')['hidden_state_storage']=='pooled_only'
    assert not (tmp_path/'run/activations').exists()


def test_both_readers_use_the_common_question_intersection_and_own_scales(tmp_path,monkeypatch):
    from steer_opd.method_config import DIRECTION_VERSION
    from steer_opd.resumable_core import atomic_torch
    from test_ts_directions import pairs
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    module=importlib.import_module('prepare_compass_banks')
    monkeypatch.setattr(module,'project_path',lambda p:tmp_path/p)
    monkeypatch.setattr(module,'checkpoint_content_id',lambda p,**kw:p.name+'_weights')
    monkeypatch.setattr(module,'initialize_stage',lambda *a:'cpu_runtime')
    cfg={'run_dir':'run','data_dir':'data','student':'S0','teacher':'T0','layers':[26],'teacher_layers':[29],
         'minimum_split_questions':1,'alpha':.1,'direction':'T-S','direction_version':DIRECTION_VERSION,
         'include_teacher_ablation':True}
    student=pairs();student['bad']=copy.deepcopy(student['a'])
    teacher=pairs()
    for record in teacher.values():
        if record['usable']:
            record['layers']['29']=record['layers'].pop('26')
            record['layers']['29']['norms']=[v*3 for v in record['layers']['29']['norms']]
    atomic_json(tmp_path/'data/questions.json',[{'id':q} for q in student])
    for role,pool,folder,layer,version in [('S0',student,'activations',26,DIRECTION_VERSION),
                                         ('T0',teacher,'activations_teacher',29,TEACHER_DIRECTION_VERSION)]:
        identity={'checkpoint':role,'checkpoint_content_id':role+'_weights','layer':layer,'direction_version':version}
        path=tmp_path/'run'/folder/'pooled.pt';atomic_torch(path,{'identity':identity,'pooled':pool})
        StageCheckpoint(path.parent,digest(identity)).commit([path])
    module.build(cfg,'student');module.build(cfg,'teacher')
    left=read_json(tmp_path/'run/banks/manifest.json');right=read_json(tmp_path/'run/banks_teacher/manifest.json')
    assert left['valid_ids']==right['valid_ids']==['a','b']
    assert float(torch.load(tmp_path/'run/banks/common_scales.pt',weights_only=True)['26'])==10
    assert float(torch.load(tmp_path/'run/banks_teacher/common_scales.pt',weights_only=True)['29'])==30


@pytest.mark.parametrize('wrong_reader',[False,True])
def test_teacher_preparation_binds_banks_to_teacher_checkpoint_not_student(tmp_path,monkeypatch,wrong_reader):
    from steer_opd import compass_job
    from steer_opd.resumable_core import atomic_torch
    from steer_opd import checkpoint_identity
    from test_compass_alignment import tiny
    s,t,plan=toy_pair()
    row={'id':'q','key':'q_0','model':'S','model_content_id':'S_weights','prompt_token_ids':[6,7],
         'token_ids':[1,2,3,0],'rollout_topk':'q_0.pt','steering':False}
    native={**row,'prompt_token_ids':[8,9],'token_ids':plan['native_response_ids']['teacher']}
    source=score_batch(tiny(),[row])[0]
    teacher={**score_batch(tiny_teacher(),[native])[0],'binding':'native_teacher'}
    pooled={'q':{'usable':True,'layers':{'29':{'pools':[torch.ones(24),torch.zeros(24)],
                                              'norms':[torch.ones(2)*10,torch.ones(2)*10]}}}}
    banks,_=make_common_banks(pooled,[29],{'checkpoint':'T','step':0,
        'checkpoint_content_id':'S_weights' if wrong_reader else 'T_weights'},direction='S-T')
    atomic_torch(tmp_path/'bank.pt',banks['S-T_global'])
    atomic_torch(tmp_path/'scores/q_0.pt',teacher);atomic_torch(tmp_path/'source/q_0.pt',source)
    monkeypatch.setattr(compass_job,'project_path',lambda p:tmp_path/p)
    monkeypatch.setattr(compass_job,'prepare_plans',lambda *a:([row],{'q_0':plan},{'teacher':[native]},{'student':s,'teacher':t}))
    monkeypatch.setattr(compass_job,'native_score_binding',lambda *a:'native_teacher')
    monkeypatch.setattr(compass_job,'load_rollout_topk',lambda *a:source)
    monkeypatch.setattr(checkpoint_identity,'checkpoint_content_id',lambda p,**kw:p.name+'_weights')
    job={'stage':'static_overlap_screen','optimizer_updates':0,'student':'S','teacher':'T','steer_role':'teacher',
         'direction_version':TEACHER_DIRECTION_VERSION,'banks':{'S-T_global':'bank.pt'},
         'conditions':[{'id':'scheme_07','direction':'S-T','steer_role':'teacher','bank':'S-T_global',
                        'gate':'static_overlap','granularity':'global','layers':[29],'alpha':.1}],
         'teacher_scores':'scores','source_dir':'source','output':'out','support_size':47,
         'threshold':.5,'identity':{'metric':METRIC},'bind_checkpoint_content':True,'require_dual_metrics':True}
    if wrong_reader:
        with pytest.raises(ValueError,match='intervention-reader checkpoint'):
            compass_job.prepare_screen(job)
    else:
        prepared=compass_job.prepare_screen(job)
        assert prepared[2].native_rows['q_0']['token_ids']==[11,12]
        assert prepared[2].source_scores['q_0'] is source


def test_historical_baseline_adapter_accepts_only_retired_direction_metadata():
    from steer_opd.paired_baseline import equivalent_source_mask
    from steer_opd.method_config import DIRECTION_VERSION
    row={'id':'q','key':'q_0','prompt_token_ids':[1,2],'token_ids':[3,4]}
    identity={'metric':METRIC,'student_checkpoint':'S','teacher_checkpoint':'T',
        'student_tokenizer_sha256':'s','teacher_tokenizer_sha256':'t','support_id':'support',
        'student_score_sha256':'score_s','teacher_score_sha256':'score_t'}
    old=freeze_static_mask(row,[.25,.75],.5,{**identity,'direction_version':DIRECTION_VERSION})
    new=freeze_static_mask(row,[.25,.75],.5,identity)
    equivalent_source_mask(old,new,row)
    assert old['mask_id']!=new['mask_id']
    for changed in (freeze_static_mask(row,[.25,.75],.6,identity),
                    freeze_static_mask(row,[.25,.75],.5,{**identity,'teacher_score_sha256':'different'}),
                    freeze_static_mask(row,[.1,.75],.5,identity)):
        with pytest.raises(ValueError,match='Historical baseline'):equivalent_source_mask(old,changed,row)
    with pytest.raises(ValueError,match='mask source'):
        equivalent_source_mask(old,new,{**row,'prompt_token_ids':[2,3]})


def test_teacher_only_supplement_selects_four_and_binds_original_baseline(launcher):
    from steer_opd.compass_campaign import scheme_conditions
    cfg=launcher.build_config(launcher.parser().parse_args(['--teacher-ablation','1','--schemes','5,6,7,8',
        '--student-baseline-run','original_run']))
    assert cfg['student_baseline_run']=='original_run'
    assert len(scheme_conditions(cfg))==4 and cfg['confirm_top_n']==4
    with pytest.raises(ValueError,match='teacher-only'):
        launcher.build_config(launcher.parser().parse_args(['--teacher-ablation','1','--student-baseline-run','original_run']))
