"""Stage-oriented overlap campaign with content-verified upstream reuse."""
import csv
import importlib
from pathlib import Path

import torch

from steer_opd import stage_contracts as keys
from steer_opd.artifact_reuse import StageArtifacts,contract
from steer_opd.checkpoint_identity import checkpoint_content_id
from steer_opd.compass_campaign import (frozen_json,generation_job,make_base_job,make_source_overlap_job,
    prepare_screen_jobs,prepare_all_training_job,role_screen_job)
from steer_opd.compass_job import (load_codecs,model_path,prepare_plans,safe_key,
    cached_source_metrics)
from steer_opd.compass_scoring import AlignedMetricAdapter
from steer_opd.direction_bank import file_sha256
from steer_opd.resumable_core import ROOT,atomic_json,digest,read_json
from steer_opd.rollout_topk import load_rollout_topk
from steer_opd.stage_io import artifact_dir,component_target,metadata_dir,project
from steer_opd.strength_schedule import confirm_with_counterparts
from steer_opd.retention import minimal


def token_metric_value(metrics):
    """Read the stored token mean, including the original overlap field name."""
    if 'token_weighted_value' in metrics:
        return metrics['token_weighted_value']
    return metrics['token_weighted_overlap16']


def data_contract(cfg,codecs):
    source=project(cfg['input_data_dir'])/'questions_source.json'
    if not source.exists():raise FileNotFoundError('Prepare a frozen INPUT_DATA_DIR before running this parameter set')
    if cfg.get('data_policy')=='all_train':
        return contract('data',{'input_sha256':file_sha256(source),'data_policy':'all_train',
            'questions':cfg['question_count'],'codecs':{r:keys.codec_key(c) for r,c in codecs.items()},
            'seed':42,'max_prompt_tokens':cfg['max_prompt_tokens'],'enable_thinking':cfg['enable_thinking']},
            code=('src/steer_opd/math_data.py','scripts/prepare_ablation_data.py','src/steer_opd/compass_campaign.py'))
    return contract('data',{'input_sha256':file_sha256(source),'split_sha256':file_sha256(source.parent/'split.json'),
        'questions':cfg['question_count'],
        'codecs':{r:keys.codec_key(c) for r,c in codecs.items()},'seed':42,
        'select_fraction':cfg['select_fraction'],'max_prompt_tokens':cfg['max_prompt_tokens'],
        'enable_thinking':cfg['enable_thinking']},
        code=('src/steer_opd/math_data.py','scripts/prepare_ablation_data.py'))


def prepare_data(cfg,codecs=None):
    codecs=load_codecs(cfg) if codecs is None else codecs
    store=StageArtifacts(cfg);science=data_contract(cfg,codecs);target=component_target(cfg,'data')
    found=store.find('data',science,target)
    if found:return store.bind(target,found,science)
    prepare=importlib.import_module('prepare_ablation_data').prepare
    prepare(cfg)
    folder=project(target)
    membership='train_ids.json' if cfg.get('data_policy')=='all_train' else 'split.json'
    return store.publish('data',science,folder,files=[folder/name for name in
        ('questions_source.json','questions.json',membership,'sampling_manifest.json')])


class Campaign:
    def __init__(self,cfg,progress,gpu,cpu):
        self.cfg=dict(cfg);self.cfg['stage_inputs']=dict(cfg.get('stage_inputs',{}))
        self.progress=progress;self.gpu=gpu;self.cpu=cpu
        self.store=StageArtifacts(cfg);self.records={}
        self.codecs=load_codecs(cfg)
        self.models={r:checkpoint_content_id(model_path(cfg[r])) for r in ('teacher','student')}
        self.runtime=keys.runtime_key(ROOT/cfg['engine_python'])
        self.cfg['_screen_runtime']=(keys.runtime_key(ROOT/cfg['screen_python'])
                                      if cfg.get('screen_python') else self.runtime)

    def remember(self,name,found):
        folder,packet=found;self.records[name]=packet
        self.cfg['stage_inputs'][name]=str(folder.relative_to(ROOT))
        if name=='data':
            self.cfg['data_dir']=str(folder.relative_to(ROOT));self.cfg['question_source']=self.cfg['data_dir']+'/questions_source.json'
        frozen_json(project(metadata_dir(self.cfg))/'upstreams'/f'{name}.json',{
            'artifact_id':packet['artifact_id'],'contract_key':packet['contract_key'],
            'descriptor':packet['descriptor_path'],'descriptor_sha256':file_sha256(project(packet['descriptor_path'])),
            'owner':packet['owner']})
        return folder

    def use(self,name,science,action,*,target=None,files=None,materialized=False):
        target=project(target or component_target(self.cfg,name))
        found=self.store.find(name,science,target,materialized=materialized)
        if found:
            self.progress.event('cache_reused',name=name,committed_units=1)
            result=self.store.bind(target,found,science)
        else:
            action(target)
            result=self.store.publish(name,science,target,files=files(target) if files else None)
        folder=self.remember(name,result)
        self.progress.update(key=name)
        return folder

    def pid(self,name):return self.records[name]['artifact_id']

    def jobfile(self,stage,name,payload):
        file=project(self.cfg['step_dirs'][str(stage)])/'metadata/jobs'/f'{name}.json'
        frozen_json(file,payload);return file

    def generation(self,role,questions):
        science=keys.generation(self.cfg,role,questions,self.codecs[role],self.models[role],self.runtime)
        def run(target):
            job=generation_job(self.cfg,self.codecs[role],role,questions,source=role=='student')
            job['output']=str(target.relative_to(ROOT))
            file=self.jobfile(1,role,job)
            self.gpu('reference_teacher' if role=='teacher' else 'source',
                self.cfg['teacher_gpus'] if role=='teacher' else 1,
                ['scripts/generate_static_sources.py','--job',str(file)],stage_output=target,stage=1)
        return self.use(role,science,run)

    def directions(self,questions):
        parents={r:self.pid(r) for r in ('teacher','student')}
        roles=('student','teacher') if self.cfg.get('include_teacher_ablation') else ('student',)
        if minimal(self.cfg) and self.reuse_direction_banks(questions,roles,parents):
            self.retire_direction_pools(roles)
            return
        for role in roles:
            science=keys.capture(self.cfg,role,self.models[role],self.codecs[role],parents,self.runtime)
            def run(target,role=role):
                file=self.jobfile(2,'capture_'+role,self.cfg)
                self.gpu('activations' if role=='student' else 'activations_teacher',
                    self.cfg['teacher_gpus'] if role=='teacher' else 1,
                    ['scripts/capture_student_activations.py','--config',str(file),'--reader',role],
                    stage_output=target,stage=2)
            self.use('capture_'+role,science,run,materialized=True)
        common={q['id'] for q in questions}
        for role in roles:
            value=torch.load(project(artifact_dir(self.cfg,'capture_'+role))/'pooled.pt',weights_only=True,map_location='cpu')
            common.intersection_update(qid for qid,r in value['pooled'].items() if r['usable'])
        ordered=[q['id'] for q in questions if q['id'] in common]
        self.cfg['allowed_question_ids']=ordered
        for role in roles:
            science=keys.bank(self.cfg,role,self.pid('capture_'+role),ordered)
            def run(target,role=role):
                file=self.jobfile(2,'bank_'+role,self.cfg)
                self.cpu('banks' if role=='student' else 'banks_teacher','scripts/prepare_compass_banks.py',file,
                         ('--reader',role),stage=2)
            self.use('bank_'+role,science,run)
        if minimal(self.cfg):self.retire_direction_pools(roles)

    def reuse_direction_banks(self,questions,roles,parents):
        """Resolve complete banks before requesting intentionally retired pools."""
        from steer_opd.sv_retention import usable_ids
        captures={};banks={};common={q['id'] for q in questions}
        for role in roles:
            science=keys.capture(self.cfg,role,self.models[role],self.codecs[role],parents,self.runtime)
            found=self.store.find('capture_'+role,science,component_target(self.cfg,'capture_'+role))
            if found is None:return False
            captures[role]=(science,found)
            common.intersection_update(usable_ids(found[0]))
        ordered=[q['id'] for q in questions if q['id'] in common]
        for role in roles:
            science=keys.bank(self.cfg,role,captures[role][1][1]['artifact_id'],ordered)
            found=self.store.find('bank_'+role,science,component_target(self.cfg,'bank_'+role))
            if found is None:return False
            banks[role]=(science,found)
        self.cfg['allowed_question_ids']=ordered
        for kind,values in [('capture',captures),('bank',banks)]:
            for role,(science,found) in values.items():
                self.remember(kind+'_'+role,self.store.bind(component_target(self.cfg,kind+'_'+role),found,science))
        return True

    def retire_direction_pools(self,roles):
        from steer_opd.sv_retention import retire_pool
        banks=[(project(self.records['bank_'+role]['owner']),self.records['bank_'+role]) for role in roles]
        for role in roles:
            packet=self.records['capture_'+role];folder=project(packet['owner'])
            # Never delete a shared upstream or a historical imported owner.
            if folder==project(component_target(self.cfg,'capture_'+role)):
                retire_pool(self.store,(folder,packet),banks)

    def source_inputs(self,source_rows):
        run=project(self.cfg['run_dir']);frozen_json(run/'source_rows.json',source_rows)
        rows_file=str((run/'source_rows.json').relative_to(ROOT))
        base=make_source_overlap_job(self.cfg,self.codecs,rows_file) if self.cfg.get('stop_after')=='source_overlap' else make_base_job(self.cfg,self.codecs,rows_file)
        science=keys.alignment(self.cfg,self.pid('student'),self.codecs)
        self.use('alignment',science,lambda _:prepare_plans(base,self.codecs),
                 files=lambda folder:list(folder.glob('*.json')))
        base['alignment_dir']=str(artifact_dir(self.cfg,'alignment'))
        base['alignment_source_dir']=base['alignment_dir']
        parents={'student':self.pid('student'),'alignment':self.pid('alignment')}
        science=keys.native(self.cfg,self.models['teacher'],self.codecs['teacher'],parents,self.runtime)
        def score(target):
            base['teacher_scores']=str(target.relative_to(ROOT))
            file=self.jobfile(3,'native_teacher',base)
            self.gpu('native_teacher',self.cfg['teacher_gpus'],
                ['scripts/score_compass_sources.py','--job',str(file),'--role','teacher','--execute'],
                stage_output=target,stage=3)
        self.use('native_teacher',science,score)
        base['teacher_scores']=str(artifact_dir(self.cfg,'native_teacher'))
        return base

    def source_metrics(self,base):
        science=keys.metrics({k:self.pid(k) for k in ('student','native_teacher','alignment')})
        def compute(target):
            base['source_metrics_dir']=str(target.relative_to(ROOT))
            rows,plans,_,_=prepare_plans(base,self.codecs)
            adapter=AlignedMetricAdapter(plans,self.codecs['student'],self.codecs['teacher'],require_dual=True)
            for row in rows:
                key=safe_key(row);source=load_rollout_topk(project(base['source_dir']),row,16,base['support_size'])
                teacher_path=project(base['teacher_scores'])/f'{key}.pt'
                teacher=torch.load(teacher_path,weights_only=True,map_location='cpu')
                cached_source_metrics(base,row,source,teacher,adapter,
                    file_sha256(project(base['source_dir'])/row['rollout_topk']),file_sha256(teacher_path))
        self.use('source_metrics',science,compute,files=lambda folder:[*folder.glob('*.pt'),*folder.glob('*.metrics.json')])
        base['source_metrics_dir']=str(artifact_dir(self.cfg,'source_metrics'))

    def screen(self,job,split):
        students=None
        rows=read_json(project(job['rows']))
        for role in ('student','teacher'):
            part=role_screen_job(job,role)
            if part is None:continue
            if role=='teacher' and students is not None:part['paired_student_baseline']=str((students/'baseline').relative_to(ROOT))
            names=('student','native_teacher','alignment','source_metrics')
            if self.cfg.get('data_policy')!='all_train':names+=('calibration',)
            parent={k:self.pid(k) for k in names}
            parent['bank']=self.pid('bank_'+role)
            if role=='teacher' and students is not None:parent['student_baseline']=self.pid('screen_'+split+'_student')
            elif role=='teacher' and self.cfg.get('student_baseline_run'):
                from steer_opd.paired_baseline import reference
                fields,parent['student_baseline']=reference(self,split)
                part.update(fields)
            science=keys.screen(self.cfg,role,part['conditions'],parent,rows)
            name='screen_'+split+'_'+role;target=project(part['output'])
            def run(target,part=part,role=role):
                file=self.jobfile(3,split+'_'+role,part)
                interpreter=str(ROOT/self.cfg['screen_python']) if self.cfg.get('screen_python') else None
                self.gpu('overlap_'+split+'_'+role,self.cfg['teacher_gpus'] if role=='teacher' else 1,
                    ['scripts/run_overlap_ablation.py','--job',str(file),'--execute'],
                    interpreter=interpreter,stage_output=target,stage=3)
            found=self.use(name,science,run,target=target)
            if role=='student':students=found
        # Reused role outputs are immutable; compose root summaries from their owners.
        summaries={}
        for role in ('student','teacher'):
            name='screen_'+split+'_'+role
            if name in self.records:summaries.update(read_json(project(self.records[name]['owner'])/'summaries.json'))
        root=project(job['output'])
        if not (root/'complete.json').exists():frozen_json(root/'summaries.json',summaries)
        return summaries


def write_summary(cfg,source_summaries,records):
    root=project(cfg['step_dirs']['6']);root.mkdir(parents=True,exist_ok=True)
    sources={k:{'artifact_id':v['artifact_id'],'owner':v['owner'],'descriptor':v['descriptor_path'],
                'descriptor_sha256':file_sha256(project(v['descriptor_path']))} for k,v in records.items()}
    frozen_json(root/'metadata/source_index.json',sources)
    frozen_json(root/'summaries.json',source_summaries)
    if not source_summaries:
        source=project(records['source_report']['owner'])/'summaries.json'
        value=read_json(source);frozen_json(root/'source_summaries.json',value)
        lines=['# 普通源overlap汇总','',f"RUN_NAME={cfg['run_name']}，PARAM_NAME={cfg['param_name']}。无SV，0次更新。",'',
               '| 划分 | 指标 | 题目数 | 题目等权 | token加权 |','|---|---|---:|---:|---:|']
        with (root/'comparison.csv').open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['split','metric','questions','question_weighted_value','token_weighted_value'])
            for split,values in [('all',value['metrics']),*((k,v) for k,v in value['by_split'].items())]:
                for metric,v in values.items():
                    writer.writerow([split,metric,v['questions'],v['question_weighted_value'],v['token_weighted_value']])
                    lines.append(f"| {split} | {metric} | {v['questions']} | {v['question_weighted_value']} | {v['token_weighted_value']} |")
        (root/'analysis.md').write_text('\n'.join(lines)+'\n')
        return
    file=root/'comparison.csv'
    columns=['split','scheme','direction','intervention_role','metric','questions','question_weighted_value',
             'question_weighted_delta_pp','token_value','token_weighted_delta_pp']
    with file.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        for split,values in source_summaries.items():
            for name,summary in values.items():
                for metric,value in summary['metrics'].items():
                    writer.writerow({'split':split,'scheme':name,'direction':summary['condition']['direction'],
                        'intervention_role':summary.get('intervention_role','student'),'metric':metric,
                        **{k:value[k] for k in columns[5:] if k!='token_value'},
                        'token_value':token_metric_value(value)})
    lines=['# Overlap结果汇总','',f"RUN_NAME={cfg['run_name']}，PARAM_NAME={cfg['param_name']}。0次参数更新。",'',
           '| 划分 | 方案 | 干预对象 | Δoverlap（pp，题目等权） | ΔP_overlap（pp，题目等权） |',
           '|---|---|---|---:|---:|']
    for split,values in source_summaries.items():
        for name,s in values.items():lines.append(f"| {split} | {name} | {s.get('intervention_role','student')} | {s['metrics']['overlap']['question_weighted_delta_pp']:+.4f} | {s['metrics']['P_overlap']['question_weighted_delta_pp']:+.4f} |")
    scope=('全部题目属于同一训练池，同时用于SV、overlap和OPD；独立评测使用MATH-500及AIME2025。'
           if cfg.get('data_policy')=='all_train' else 'Confirm是训练侧复核，不能视为独立测试。')
    lines+=['',scope+'教师干预的增益表示教师适配，不代表学生学习。',
            '来源与SHA身份见metadata/source_index.json；完整分布与SV保留在来源阶段，不重复复制。','']
    (root/'analysis.md').write_text('\n'.join(lines))


def execute(cfg,progress,gpu,cpu):
    campaign=Campaign(cfg,progress,gpu,cpu)
    campaign.remember('data',prepare_data(cfg,campaign.codecs));progress.update(key='data')
    questions=read_json(project(campaign.cfg['data_dir'])/'questions.json')
    if cfg.get('stop_after')!='source_overlap':campaign.generation('teacher',questions)
    source=campaign.generation('student',questions)
    rows=[read_json(source/f"{q['id']}_0.json") for q in questions]
    if cfg.get('stop_after')!='source_overlap':campaign.directions(questions)
    base=campaign.source_inputs(rows);campaign.source_metrics(base)
    if cfg.get('data_policy')=='all_train':
        job=prepare_all_training_job(campaign.cfg,base,campaign.codecs)
        summary=campaign.screen(job,'train')
        write_summary(cfg,{'train':summary},campaign.records)
        progress.update(key='summary');progress.total=progress.done
        from steer_opd.experiment_runtime import export_artifacts
        export_artifacts(campaign.cfg,project(cfg['run_dir']))
        atomic_json(project(cfg['run_dir'])/'campaign.json',{'status':'awaiting_user_review',
            'parameter_updates':0,'data_policy':'all_train','training_questions':len(questions),
            'evaluated_ids':[c['id'] for c in job['conditions']],'automatic_opd':False,
            'step_dirs':cfg['step_dirs'],'summary_dir':cfg['step_dirs']['6']})
        return
    if cfg.get('stop_after')=='source_overlap':
        from steer_opd.source_overlap import save_source_overlap
        science=contract('source_report',{'split_sha256':file_sha256(project(campaign.cfg['data_dir'])/'split.json')},
            parents={k:campaign.pid(k) for k in ('source_metrics','data')},
            code=('src/steer_opd/source_overlap.py',))
        campaign.use('source_report',science,lambda _:save_source_overlap(campaign.cfg,base,campaign.codecs))
        atomic_json(project(cfg['run_dir'])/'campaign.json',{'status':'awaiting_overlap_review','parameter_updates':0,
                    'sv_injection':False,'automatic_opd':False})
        write_summary(cfg,{},campaign.records);progress.update(key='summary')
        from steer_opd.experiment_runtime import export_artifacts
        export_artifacts(campaign.cfg,project(cfg['run_dir']));return
    split=read_json(project(campaign.cfg['data_dir'])/'split.json')
    selected=set(split['select_ids']).intersection(campaign.cfg['allowed_question_ids'])
    science=keys.calibration(campaign.cfg,{'metrics':campaign.pid('source_metrics')},selected)
    folder=campaign.store.find('calibration',science,component_target(cfg,'calibration'))
    if folder:campaign.remember('calibration',campaign.store.bind(component_target(cfg,'calibration'),folder,science))
    select_job,confirm_job=prepare_screen_jobs(campaign.cfg,base,campaign.codecs)
    if folder is None:
        path=project(artifact_dir(campaign.cfg,'calibration'))
        campaign.remember('calibration',campaign.store.publish('calibration',science,path,
            files=[path/'thresholds.json',path/'threshold.json']))
    progress.update(key='calibration')
    select=campaign.screen(select_job,'select')
    ranked=sorted(select,key=lambda k:(-select[k]['question_weighted_delta_pp'],k))
    chosen=confirm_with_counterparts(ranked,select_job['conditions'],cfg['confirm_top_n'])
    frozen_json(project(cfg['run_dir'])/'selected_schemes.json',{'ids':chosen,'selection_uses_confirm':False,
        'parameter_updates':0,'baseline_is_best_on_select':select[ranked[0]]['question_weighted_delta_pp']<=0})
    confirm_job['conditions']=[c for c in confirm_job['conditions'] if c['id'] in chosen]
    confirm=campaign.screen(confirm_job,'confirm')
    write_summary(cfg,{'select':select,'confirm':confirm},campaign.records)
    progress.update(key='summary')
    progress.total=progress.done
    from steer_opd.experiment_runtime import export_artifacts
    export_artifacts(campaign.cfg,project(cfg['run_dir']))
    atomic_json(project(cfg['run_dir'])/'campaign.json',{'status':'awaiting_user_review','parameter_updates':0,
        'select_schemes':len(select),'confirm_ids':chosen,'automatic_opd':False,
        'step_dirs':cfg['step_dirs'],'summary_dir':cfg['step_dirs']['6']})
