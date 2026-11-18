"""CPU-only overlap reports for the original unsteered student trajectories."""
import csv

import torch

from steer_opd.compass_job import prepare_plans, project_path, native_score_binding, model_path, SCORING_ROLES, cached_source_metrics
from steer_opd.compass_scoring import AlignedMetricAdapter
from steer_opd.direction_bank import file_sha256
from steer_opd.experiment_runtime import initialize_stage, Progress
from steer_opd.tensor_checkpoint import StageCheckpoint
from steer_opd.resumable_core import atomic_json, digest, read_json
from steer_opd.rollout_topk import load_rollout_topk
from steer_opd.static_masks import validate_native_score


METRICS=('overlap','P_overlap')


def summarize(rows, records):
    result={}
    for name in METRICS:
        per_question={}
        for row in rows:
            value=records[row['key']][name]
            per_question.setdefault(row['id'],[]).append(value[torch.isfinite(value)])
        values={qid:torch.cat(parts) for qid,parts in per_question.items()}
        values={qid:v for qid,v in values.items() if len(v)}
        positions=sum(len(v) for v in values.values())
        result[name]={'questions':len(values),'valid_positions':positions,
            'question_weighted_value':sum(float(v.mean()) for v in values.values())/len(values) if values else None,
            'token_weighted_value':sum(float(v.sum()) for v in values.values())/positions if positions else None,
            'per_question_mean':{qid:float(v.mean()) for qid,v in values.items()},
            'coverage':positions/sum(len(r['token_ids']) for r in rows) if rows else 0.}
    return result


def plot_source(rows, records, output, requested_keys):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    by_key={r['key']:r for r in rows}
    files=[]
    for metric in METRICS:
        folder=output/metric;folder.mkdir(parents=True,exist_ok=True)
        with PdfPages(folder/'first3.pdf') as pdf:
            for key in requested_keys:
                if key not in by_key:continue
                record=records[key];y=record[metric].double().cpu().numpy()
                csv_path=folder/f'{key}.csv'
                with csv_path.open('w',newline='') as stream:
                    writer=csv.writer(stream)
                    writer.writerow(['prediction_idx',metric,'metric_valid','split'])
                    writer.writerows((i,v,int(torch.isfinite(record[metric][i])),record['split']) for i,v in enumerate(y))
                fig,axis=plt.subplots(figsize=(12,3))
                axis.plot(range(len(y)),y,color='#2563eb',lw=.8)
                axis.set(xlabel='Student response prediction index',ylabel=metric,ylim=(-.02,1.02),
                    title=f'{key}: original no-SV rollout; {record["split"]}; zero updates')
                fig.tight_layout()
                for suffix in ('png','svg'):fig.savefig(folder/f'{key}.{suffix}',dpi=160)
                pdf.savefig(fig);plt.close(fig)
        ordered=[r['key'] for r in rows]
        means=[float(records[k][metric][torch.isfinite(records[k][metric])].mean()) for k in ordered]
        fig,axis=plt.subplots(figsize=(12,3))
        for split,color in [('select','#2563eb'),('confirm','#ea580c')]:
            indices=[i for i,k in enumerate(ordered) if records[k]['split']==split]
            axis.scatter(indices,[means[i] for i in indices],s=14,color=color,label=split)
        axis.set(xlabel='Original source manifest order',ylabel=f'Mean {metric}',ylim=(-.02,1.02),
            title='Unsteered source overlap by question; no candidate selection')
        axis.legend();fig.tight_layout()
        for suffix in ('png','svg','pdf'):fig.savefig(folder/f'question_means.{suffix}',dpi=160)
        plt.close(fig)
        with (folder/'question_means.csv').open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['key','question_id','split',f'mean_{metric}'])
            writer.writerows((k,by_key[k]['id'],records[k]['split'],v) for k,v in zip(ordered,means))
        files.extend(sorted(p for p in folder.iterdir() if p.suffix in ('.png','.svg','.pdf','.csv')))
    manifest=output/'plots.json'
    atomic_json(manifest,{'source':'original_unsteered_rollout','requested_keys':requested_keys,
        'unavailable_keys':[k for k in requested_keys if k not in by_key],
        'files':{str(p.relative_to(output)):file_sha256(p) for p in files}})
    return [*files,manifest]


def save_source_overlap(cfg,job,codecs=None):
    if job.get('stage')!='source_overlap' or job.get('optimizer_updates')!=0 or cfg.get('stop_after')!='source_overlap':
        raise ValueError('Expected a source-only, zero-update overlap report')
    output=project_path(cfg['run_dir'])/'source_overlap'
    runtime=initialize_stage(job,output)
    rows,plans,role_rows,codecs=prepare_plans(job,codecs)
    native_rows={role:{r['key']:r for r in values} for role,values in role_rows.items()}
    split=read_json(project_path(cfg['data_dir'])/'split.json')
    membership={qid:name for name in ('select','confirm') for qid in split[name+'_ids']}
    input_paths=[project_path(job['rows']),project_path(cfg['data_dir'])/'split.json',
        project_path(job['source_dir'])/'complete.json',
        *(project_path(job[f'{role}_scores'])/'complete.json' for role in SCORING_ROLES)]
    inputs={str(p):file_sha256(p) for p in input_paths}
    stage=StageCheckpoint(output,digest({'job':job,'runtime':runtime,'inputs':inputs}))
    with Progress(output,'source_overlap_report',len(rows),unit='question',show_bar=False) as progress:
        if stage.load() is not None:progress.resume(len(rows));return
        adapter=AlignedMetricAdapter(plans,codecs['student'],codecs['teacher'],require_dual=True)
        student_content=None
        if job.get('bind_checkpoint_content'):
            from steer_opd.checkpoint_identity import checkpoint_content_id
            student_content=checkpoint_content_id(model_path(job['student']))
        records={}
        for row in rows:
            key=row['key']
            if row.get('model')!=job['student'] or row['id'] not in membership:
                raise ValueError('Source model/question differs from the frozen experiment')
            if student_content is not None and row.get('model_content_id')!=student_content:
                raise ValueError('Source checkpoint contents changed')
            student=load_rollout_topk(project_path(job['source_dir']),row,16,job['support_size'])
            native={}
            for role in SCORING_ROLES:
                value=torch.load(project_path(job[f'{role}_scores'])/f'{key}.pt',map_location='cpu',weights_only=True)
                own_row=native_rows[role][key]
                if value.get('binding')!=native_score_binding(job,role,own_row,codecs[role]):
                    raise ValueError('Source probability cache binding changed')
                validate_native_score(value,len(own_row['token_ids']),codecs[role].support_size)
                native[role]=value
            metrics=cached_source_metrics(job,row,student,native['teacher'],adapter,
                file_sha256(project_path(job['source_dir'])/row['rollout_topk']),
                file_sha256(project_path(job['teacher_scores'])/f'{key}.pt'))
            records[key]={**metrics,'question_id':row['id'],'split':membership[row['id']],
                'source':'original_unsteered_rollout','parameter_updates':0,'sv_injection':False}
            del student,native,value
            progress.update(key=key)
        summary={'scope':'original_unsteered_source_no_candidate_comparison','parameter_updates':0,'sv_injection':False,
            'metrics':summarize(rows,records),'by_split':{name:summarize([r for r in rows if membership[r['id']]==name],records)
                for name in ('select','confirm')},
            'input_requests':len(read_json(project_path(job['rows']))),'aligned_requests':len(rows),
            'rejected':read_json(project_path(job.get('alignment_source_dir',job['alignment_dir']))/'manifest.json')['rejected']}
        atomic_json(output/'summaries.json',summary)
        requested_keys=[r['key'] for r in read_json(project_path(job['rows']))[:3]]
        plots=plot_source(rows,records,output,requested_keys)
        stage.save_tensors(records,[output/'summaries.json',*plots])
        progress.mark_committed(len(rows))
