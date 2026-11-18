#!/usr/bin/env python3
"""Continuous native teacher prefix scoring; commit only the complete batch."""
import argparse

from steer_opd.compass_job import prepare_plans, project_path, model_path, native_score_binding, SCORING_ROLES
from steer_opd.compass_scoring import load_text_scorer
from steer_opd.execution_policy import require_execution_allowed
from steer_opd.resumable_core import digest, read_json
from steer_opd.static_screen import score_batch
from steer_opd.experiment_runtime import initialize_stage, Progress, require_artifact_space
from steer_opd.tensor_checkpoint import StageCheckpoint


def score(job,role):
    if role not in SCORING_ROLES:raise ValueError('Only teacher scoring is used by this experiment')
    require_execution_allowed(job)
    output=project_path(job[f'{role}_scores']);runtime=initialize_stage(job,output)
    _,_,role_rows,codecs=prepare_plans(job);rows=role_rows[role];codec=codecs[role]
    stage=StageCheckpoint(output,digest({'job':job,'role':role,'runtime':runtime}))
    with Progress(output,f'native_{role}',len(rows),show_bar=False) as progress:
        if stage.load() is not None:progress.resume(len(rows));return
        require_artifact_space(output,sum(len(r['token_ids']) for r in rows)*(16*8+4))
        backend=job.get('probability_backend','hf_batched_static');extra=[]
        if backend=='easysteer':
            from steer_opd.easysteer_static import create_engine,fixed_prefix_scores
            model=create_engine(model_path(job[role]),codec.support_size,
                max_model_len=max(len(r['prompt_token_ids'])+len(r['token_ids']) for r in rows)+1,
                tensor_parallel_size=job.get('teacher_gpus',2) if role=='teacher' else 1,
                gpu_memory_utilization=job.get('gpu_memory_utilization',.75),seed=42,progress=progress)
            with progress.phase_scope('continuous_prefix_scoring'):
                scores=fixed_prefix_scores(model,rows,[False]*len(rows),include_sampled=True,progress=progress)
        elif backend=='hf_batched_static':
            model=load_text_scorer(model_path(job[role]),codec,codecs['student'])
            batch=job.get(f'{role}_batch_size',1)
            if batch<1:raise ValueError('Invalid reference scoring batch size')
            scores=[]
            for start in range(0,len(rows),batch):scores.extend(score_batch(model,rows[start:start+batch],chunk=job.get('chunk',128)))
        else:raise ValueError('Unknown probability backend; no automatic fallback')
        progress.finished_call(len(rows))
        with progress.phase_scope('save_complete_batch'):
            records={r['key']:{**s,'binding':native_score_binding(job,role,r,codec),'alignment_id':r['alignment_id'],
                'role':role,'checkpoint':job[role],'support_size':codec.support_size,'tokenizer_mapping_sha256':codec.mapping_sha256}
                for r,s in zip(rows,scores)}
            stage.save_tensors(records,extra)
        progress.mark_committed(len(rows))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--job',required=True)
    parser.add_argument('--role',choices=SCORING_ROLES,required=True);parser.add_argument('--execute',action='store_true')
    args=parser.parse_args();job=read_json(args.job)
    if args.execute:require_execution_allowed(job)
    if job.get('optimizer_updates')!=0:raise ValueError('Source scoring must not update parameters')
    if args.execute:score(job,args.role)
    else:
        _,_,rows,_=prepare_plans(job);print(f'CPU alignment complete; {len(rows[args.role])} {args.role} prefixes prepared')


if __name__=='__main__':main()
