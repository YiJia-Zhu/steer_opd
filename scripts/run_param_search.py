#!/usr/bin/env python3
"""Serial S-T/all/global layer/operator search on frozen prefixes; no OPD updates."""
import argparse
import csv
import fcntl
import itertools
import os
import signal
import shutil

from steer_opd.artifact_reuse import StageArtifacts
from steer_opd.compass_campaign import frozen_json
from steer_opd.direction_bank import file_sha256
from steer_opd.experiment_runtime import console_log,source_revision,timestamp
from steer_opd.resumable_core import ROOT,atomic_json,read_json
from steer_opd.stage_io import configure_layout,project
from steer_opd.stage_campaign import token_metric_value
from launch_ablation import run_command


def search_plan(cfg):
    settings=cfg['parameter_search']
    if cfg['question_count']!=1000:raise ValueError('Current parameter search fixes the 1000-question pool')
    if settings['scheme_ids']!=[5] or settings['stop_after']!='overlap':
        raise ValueError('This search only evaluates S-T/all/global scheme 5 through overlap')
    if settings.get('additive_scaling')!='raw_mean_difference_v1':raise ValueError('Declare raw additive semantics')
    if not settings['alphas'] or any(type(a) not in (int,float) or not 0<=a<=1 for a in settings['alphas']):raise ValueError('Invalid strength grid')
    if not settings['teacher_layers'] or any(type(l) is not int or not 0<=l<40 for l in settings['teacher_layers']):raise ValueError('Invalid teacher layers')
    if set(settings['operators'])!={'slerp','additive'}:raise ValueError('Search Slerp and additive explicitly')
    combinations=list(itertools.product(settings['teacher_layers'],settings['operators'],settings['alphas']))
    if len(set(combinations))!=len(combinations):raise ValueError('Duplicate grid entries')
    references={(r['teacher_layer'],r['operator'],r['alpha']):r['run'] for r in settings['completed_references']}
    if len(references)!=len(settings['completed_references']) or set(references)-set(combinations):
        raise ValueError('Invalid completed reference points')
    base=configure_layout({**cfg,'run_name':'param','param_name':settings['search_name']})
    cases=[]
    for layer,operator,alpha in combinations:
        param=f"n{cfg['question_count']}_{operator}{alpha:g}_S26_T{layer}_seed42_st_all_global_v2"
        child=configure_layout({**cfg,'run_name':'param','param_name':param})
        reuse=references.get((layer,operator,alpha))
        cases.append({'operator':operator,'alpha':alpha,'teacher_layer':layer,'param_name':param,
            'source_run':reuse or child['run_dir'],
            'kind':'completed_reference' if reuse else 'new_parameter_run',
            'scheme_ids':['scheme_05'],'gate_target_rate':settings['gate_target_rate'],
            'input_data_dir':cfg['input_data_dir']})
    # Validate/reference both existing points before spending GPU time on new ones.
    cases.sort(key=lambda c:c['kind']!='completed_reference')
    return base,cases


def completed(case, *, full=False):
    run=project(case['source_run'])
    if not (run/'campaign.json').exists():return False
    if read_json(run/'campaign.json').get('status')!='awaiting_user_review':return False
    source_cfg=read_json(run/'metadata/launch_config.json')
    if (source_cfg['question_count']!=1000 or source_cfg['sv_max_context_len']!=2048
            or source_cfg['gate_target_rate']!=case['gate_target_rate']
            or source_cfg['gate_metric']!='overlap' or source_cfg['rank_metric']!='overlap'
            or source_cfg['seed']!=42 or source_cfg['optimizer_updates']!=0
            or source_cfg.get('operator','slerp')!=case['operator']):
        raise ValueError('Reference/candidate data, operator or gating protocol changed')
    expected=project(case['input_data_dir'])
    actual=project(source_cfg['input_data_dir'])
    for name in ('questions_source.json','split.json'):
        if file_sha256(expected/name)!=file_sha256(actual/name):raise ValueError('Frozen reference question pool or split changed')
    cache=StageArtifacts({'output_root':'outputs'})
    for split in ('select','confirm'):
        folder=run/split/'teacher'
        if not (folder/'complete.json').exists():return False
        packet=read_json(folder/'summaries.json')
        names=set(case['scheme_ids'])
        if not names<=set(packet):raise ValueError('Incomplete parameter scheme set')
        if case['kind']=='new_parameter_run' and set(packet)!=names:raise ValueError('Unexpected extra parameter schemes')
        for name in names:
            value=packet[name];c=value['condition']
            if (c['alpha']!=case['alpha'] or c['layers']!=[case['teacher_layer']] or c['direction']!='S-T'
                    or c['operator']!=case['operator'] or c['gate']!='all' or c['granularity']!='global'
                    or c['steer_role']!='teacher' or c['strength_policy']!='fixed'):
                raise ValueError('Reference/candidate parameters do not match the search case')
            if case['operator']=='additive' and c.get('additive_scaling')!='raw_mean_difference_v1':raise ValueError('Additive scaling changed')
            for metric in ('overlap','P_overlap'):
                if value['metrics'][metric]['questions']!=(700 if split=='select' else 300):raise ValueError('Incomplete question count')
        if full:cache.validate(folder/'metadata/artifact.json')
    return True


def aggregate(folder,cases, *, final=False):
    sources=[];rows=[]
    for case in cases:
        if not completed(case):continue
        run=project(case['source_run']);item={**case,'splits':{}}
        for split in ('select','confirm'):
            source=run/split/'teacher/summaries.json';packet=read_json(source)
            item['splits'][split]={'path':str(source.relative_to(ROOT)),'sha256':file_sha256(source),
                'descriptor_sha256':file_sha256(source.parent/'metadata/artifact.json')}
            for name in case['scheme_ids']:
                value=packet[name]
                for metric,m in value['metrics'].items():
                    rows.append({'operator':case['operator'],'alpha':case['alpha'],'teacher_layer':case['teacher_layer'],'scheme':name,
                        'gate':value['condition']['gate'],'granularity':value['condition']['granularity'],
                        'split':split,'metric':metric,'questions':m['questions'],
                        'value':m['question_weighted_value'],'delta_pp':m['question_weighted_delta_pp'],
                        'token_value':token_metric_value(m),'token_delta_pp':m['token_weighted_delta_pp'],
                        'origin':case['kind']})
        sources.append(item)
    folder.mkdir(parents=True,exist_ok=True)
    atomic_json(folder/'metadata/source_index.json',{'cases':sources,'parameter_updates':0,
        'primary_ranking_split':'select','rank_metric':'overlap','confirm_used_for_ranking':False,
        'status':'complete' if final else 'partial'})
    if not rows:return
    with (folder/'comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    values=[r for r in rows if r['split']=='select' and r['metric']=='overlap']
    winner=max(values,key=lambda r:(r['delta_pp'],-r['alpha'],-r['teacher_layer'],r['operator']))
    atomic_json(folder/'selected_parameters.json',{'status':'final' if final else 'provisional',
        'selection_metric':'select_question_weighted_overlap_delta_pp','confirm_used_for_selection':False,
        'winners':{'scheme_05':winner}})
    lines=['# S−T／all／global参数实验汇总','',
        '固定1000题、700／300划分及源前缀，SV使用全部1000题，0次OPD更新。',
        '层29的Slerp .05／.1引用已完成结果；其余为新计算。只按Select题目等权Δoverlap排名，Confirm为内部复核。',
        '直接加法为h+alpha×原始global均值差分，Slerp的alpha为旋转比例。','',
        '| 方式 | 强度 | 教师层 | Select Δoverlap（pp） | Confirm Δoverlap（pp） | Select ΔP_overlap（pp） | Confirm ΔP_overlap（pp） |',
        '|---|---:|---:|---:|---:|---:|---:|']
    lookup={(r['operator'],r['alpha'],r['teacher_layer'],r['split'],r['metric']):r['delta_pp'] for r in rows}
    for case in sources:
        vals=[lookup[case['operator'],case['alpha'],case['teacher_layer'],split,metric]
            for metric in ('overlap','P_overlap') for split in ('select','confirm')]
        lines.append(f"| {case['operator']} | {case['alpha']:g} | {case['teacher_layer']} | "+' | '.join(f'{v:+.4f}' for v in vals)+' |')
    lines+=['','本表是教师分布适配，不代表学生训练收益；来源SHA见metadata/source_index.json。','']
    (folder/'analysis.md').write_text('\n'.join(lines))


def storage_required(settings):
    """Budget baseline+repeat+one candidate, using measured historical tensor sizes."""
    reference=project(settings['reference_run'])
    raw=sum(p.stat().st_size for split in ('select','confirm')
        for name in ('baseline','teacher_unsteered_replay','scheme_05')
        for p in (reference/split/'teacher'/name).glob('*.pt'))
    if raw<=0:raise ValueError('Missing storage estimation reference')
    masks=sum(p.stat().st_size for split in ('select','confirm') for p in (reference/split/'teacher/masks').glob('*.json'))
    return int(raw*.65)+masks+512*2**20+2**30


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--run',action='store_true');mode.add_argument('--check',action='store_true');mode.add_argument('--status',action='store_true')
    parser.add_argument('--gpu-ids',default='2,3');parser.add_argument('--config',default='configs/ablation.json')
    args=parser.parse_args();cfg=read_json(project(args.config));base,cases=search_plan(cfg)
    root=project(base['step_dirs']['6']);settings=cfg['parameter_search']
    if args.status:
        path=root/'search_progress.json';print(read_json(path) if path.exists() else {'status':'not_started'});return
    print({'output':str(root),'gpu_pool':args.gpu_ids,'cases':cases,'parameter_updates':0},flush=True)
    if not args.run:return
    root.mkdir(parents=True,exist_ok=True)
    lock=(root/'search.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('This parameter search already has a live supervisor') from None
    revision=source_revision()[0]
    request={'settings':settings,'cases':cases,'template_sha256':file_sha256(project(args.config)),
             'source_revision':revision,'sv_questions':1000,'parameter_updates':0}
    frozen_json(root/'metadata/search_request.json',request)
    def interrupted(signum,frame):raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    state={'status':'running','pid':os.getpid(),'started_at':timestamp(),'total':len(cases),'completed':0,
           'gpu_pool':args.gpu_ids,'parameter_updates':0,'new_total':sum(c['kind']=='new_parameter_run' for c in cases)}
    atomic_json(root/'search_progress.json',state)
    with console_log(root/'logs/search.log'):
        try:
            for case in cases:
                if case['kind']=='completed_reference' and not completed(case,full=True):
                    raise ValueError('A required completed reference is unavailable')
            aggregate(root,cases)
            for case in cases:
                if file_sha256(project(args.config))!=request['template_sha256'] or source_revision()[0]!=revision:
                    raise ValueError('Code/template changed during search; do not mix identities')
                state.update(current=case,updated_at=timestamp());atomic_json(root/'search_progress.json',state)
                if completed(case,full=True):
                    print(f"Reuse complete: {case['source_run']} (scheme_05 only)",flush=True)
                else:
                    env={**os.environ,'RUN_NAME':'param','PARAM_NAME':case['param_name'],
                        'TEACHER_ABLATION':'1','SCHEME_IDS':'5','CONFIRM_TOP_N':'1',
                        'GPU_IDS':args.gpu_ids,'OPERATOR':case['operator'],'ALPHA':str(case['alpha']),'TEACHER_LAYER':str(case['teacher_layer']),
                        'GATE_TARGET_RATE':str(settings['gate_target_rate']),'GATE_METRIC':'overlap','RANK_METRIC':'overlap',
                        'QUESTION_COUNT':'1000','SV_MAX_CONTEXT_LEN':'2048','STOP_AFTER':'panel','HF_PARITY':'0',
                        'INPUT_DATA_DIR':cfg['input_data_dir'],
                        'STUDENT_BASELINE_RUN':settings['student_baseline_run'],
                        'CANDIDATE_TENSOR_FORMAT':os.environ.get('CANDIDATE_TENSOR_FORMAT','parquet_zstd')}
                    required=storage_required(settings);available=shutil.disk_usage(root).free
                    state['storage_check']={'required_bytes':required,'available_bytes':available}
                    atomic_json(root/'search_progress.json',state)
                    if available<required:raise OSError(28,'Not enough destination quota for the next complete parameter point; keep committed results')
                    print(f"Start overlap: {case['operator']} strength={case['alpha']}, teacher layer={case['teacher_layer']}",flush=True)
                    run_command(['bash','scripts/run_ablation.sh','--run'],env)
                    if not completed(case,full=True):raise ValueError('Child exited without complete overlap results')
                state['completed']+=1;state['updated_at']=timestamp();atomic_json(root/'search_progress.json',state);aggregate(root,cases)
            aggregate(root,cases,final=True)
            state.update(status='complete',finished_at=timestamp());atomic_json(root/'search_progress.json',state)
        except BaseException as error:
            state.update(status='interrupted_resumable' if isinstance(error,KeyboardInterrupt) else 'failed_resumable',error=repr(error),finished_at=timestamp())
            atomic_json(root/'search_progress.json',state);raise


if __name__=='__main__':main()
