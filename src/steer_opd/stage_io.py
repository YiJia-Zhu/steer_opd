"""Output-layout specification: experiment category and parameter set are separate."""
from pathlib import Path
import re

from steer_opd.resumable_core import ROOT

STEPS={0:'data',1:'rollout',2:'sv',3:'overlap',4:'train',5:'eval',6:'summary'}


def frozen_data_dir(count=1000):
    return (f'outputs/step_0_data/ablation/T_Qwen3.5-35B-A3B_S_Qwen3-4B/'
            f'OpenThoughts3-1.2M/shared/n{count}_slerp0.1_layer26_seed42_v1')


def safe_name(value):
    if not isinstance(value,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',value) or value in ('.','..'):
        raise ValueError(f'Unsafe output name: {value!r}')
    return value


def default_param_name(cfg):
    layer=(f"S{cfg['layers'][0]}_T{cfg['teacher_layers'][0]}" if cfg.get('include_teacher_ablation')
           else f"layer{cfg['layers'][0]}")
    label=f"n{cfg['question_count']}_{cfg.get('operator','slerp')}{cfg['alpha']:g}_{layer}_seed{cfg['seed']}"
    for key,default,tag in [('gate_metric','overlap','gate'),('rank_metric','overlap','rank'),('gate_target_rate',.2,'rate'),
        ('reference_max_tokens',12288,'Tgen'),('source_max_tokens',7168,'Sgen'),
        ('sv_max_context_len',2048,'ctx'),('reference_temperature',.7,'Ttemp'),('reference_top_p',.95,'Tp')]:
        value=cfg.get(key,default)
        if value!=default:label+=f'_{tag}{value:g}' if isinstance(value,(int,float)) else f'_{tag}{value}'
    return label+'_v1'


def layout_paths(cfg):
    category=safe_name(cfg.get('run_name','ablation'))
    if category not in ('ablation','param','compar'):
        raise ValueError('RUN_NAME is the experiment category: ablation, param or compar; use PARAM_NAME for settings')
    pair=safe_name('T_'+Path(cfg['teacher']).name+'_S_'+Path(cfg['student']).name)
    train=safe_name(cfg.get('training_dataset','OpenThoughts3-1.2M'))
    param=safe_name(cfg.get('param_name') or default_param_name(cfg))
    base=Path(cfg.get('output_root','outputs'))
    absolute=base if base.is_absolute() else ROOT/base
    if not absolute.resolve().is_relative_to(ROOT.resolve()):raise ValueError('Outputs must stay inside the project')
    eval_set=safe_name(cfg.get('evaluation_dataset','AIME2025'))
    return {str(i):str(base/f'step_{i}_{name}'/category/pair/train/(eval_set if i==5 else 'shared')/param)
            for i,name in STEPS.items()}


def configure_layout(cfg):
    cfg=dict(cfg);cfg['layout_version']=1
    cfg.setdefault('run_name','ablation');cfg.setdefault('training_dataset','OpenThoughts3-1.2M')
    cfg['param_name']=cfg.get('param_name') or default_param_name(cfg)
    cfg['step_dirs']=layout_paths(cfg)
    cfg['data_dir']=cfg['step_dirs']['0'];cfg['question_source']=cfg['data_dir']+'/questions_source.json'
    cfg['run_dir']=cfg['step_dirs']['3']
    cfg.setdefault('input_data_dir',frozen_data_dir())
    cfg['stage_inputs']={}
    return cfg


def component_target(cfg, component):
    steps=cfg['step_dirs']
    if component=='data':return Path(steps['0'])
    if component in ('teacher','student'):return Path(steps['1'])/component
    if component.startswith('capture_') or component.startswith('bank_'):
        kind,role=component.split('_',1)
        return Path(steps['2'])/('reader_'+role)/('activations' if kind=='capture' else 'banks')
    paths={'native_teacher':'native_scores/teacher','alignment':'alignment','source_metrics':'source_metrics',
           'calibration':'calibration','source_report':'source_overlap','acceptance':'metadata/acceptance'}
    if component in paths:return Path(steps['3'])/paths[component]
    raise ValueError(f'Unknown reusable component: {component}')


def artifact_dir(cfg, component):
    value=cfg.get('stage_inputs',{}).get(component)
    if value is not None:return Path(value)
    if cfg.get('layout_version')==1:return component_target(cfg,component)
    # Low-level callers may supply explicit roots without a campaign layout.
    run=Path(cfg['run_dir'])
    defaults={'teacher':run/'references/teacher','student':run/'source',
        'capture_student':run/'activations','capture_teacher':run/'activations_teacher',
        'bank_student':run/'banks','bank_teacher':run/'banks_teacher',
        'native_teacher':run/'native_scores/teacher','alignment':run/'alignment',
        'source_metrics':run/'source_metrics','calibration':run,'data':Path(cfg.get('data_dir','data'))}
    return defaults[component]


def metadata_dir(cfg):
    return Path(cfg['run_dir'])/'metadata' if cfg.get('layout_version')==1 else Path(cfg['run_dir'])


def logs_dir(cfg):
    return Path(cfg['run_dir'])/'logs' if cfg.get('layout_version')==1 else Path(cfg['run_dir'])


def project(value):
    from steer_opd.artifact_paths import relocate_path
    p=Path(value);p=relocate_path(p if p.is_absolute() else ROOT/p)
    if not p.resolve().is_relative_to(ROOT.resolve()):raise ValueError('Artifact path escapes the project')
    return p
