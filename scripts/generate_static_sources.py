#!/usr/bin/env python3
"""Continuous native EasySteer generation with a whole-batch resume boundary."""
import argparse

from steer_opd.compass_job import load_codecs, model_path, project_path
from steer_opd.easysteer_static import create_engine
from steer_opd.execution_policy import require_default_scheduler_options, require_execution_allowed
from steer_opd.experiment_runtime import initialize_stage, Progress, StageCheckpoint, require_artifact_space
from steer_opd.resumable_core import atomic_json, read_json, digest
from steer_opd.rollout_topk import commit_topk_response, extract_topk


def generate(job):
    require_execution_allowed(job)
    require_default_scheduler_options(job)
    if any(r.get('seed') != 42 for r in job['requests']):
        raise ValueError('Every request must use seed=42')
    output=project_path(job['output'])
    if job.get('reuse_committed_texts'):
        from steer_opd.compass_campaign import validate_teacher_text_reuse
        count=validate_teacher_text_reuse(job)
        with Progress(output,'reference_teacher',count,show_bar=False) as progress:
            progress.resume(count)
        return
    if job.get('reuse_committed_rollout'):
        from steer_opd.compass_campaign import validate_source_rollout_reuse
        count=validate_source_rollout_reuse(job)
        with Progress(output,'source',count,show_bar=False) as progress:
            progress.resume(count)
        return
    # Historical receipts can be verified read-only under their original
    # settings. Every new engine call must bind its effective penalty.
    from steer_opd.generation_policy import presence_penalty
    penalty=presence_penalty(job,require_explicit=True)
    if job.get('bind_checkpoint_content'):
        from steer_opd.checkpoint_identity import checkpoint_content_id
        job={**job,'model_content_id':checkpoint_content_id(model_path(job['model']),receipt_dir=output/'provenance/checkpoints')}
    runtime=initialize_stage(job,output)
    stage=StageCheckpoint(output,digest({'job':job,'runtime':runtime}))
    requests=job['requests']
    with Progress(output,'source' if job.get('save_rollout_top_k') else f"reference_{job['role']}",len(requests),show_bar=False) as progress:
        if stage.load() is not None:
            progress.resume(len(requests));return
        job_hash=digest(job)
        atomic_json(output/'job.json',{'job':job,'hash':job_hash})
        codecs=load_codecs(job);codec=codecs[job['role']]
        source=bool(job.get('save_rollout_top_k'))
        engine=create_engine(model_path(job['model']),codec.support_size,
            max_model_len=job['max_tokens']+max(len(r['prompt_token_ids']) for r in requests),
            tensor_parallel_size=job.get('tensor_parallel_size',1),
            gpu_memory_utilization=job['gpu_memory_utilization'],
            batch_invariant=job['role']=='student',
            prefix_caching=job.get('enable_prefix_caching',True),seed=42,progress=progress)
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind
        params=[SamplingParams(n=1,temperature=job['temperature'],top_p=job['top_p'],top_k=-1,
            presence_penalty=penalty,
            max_tokens=job['max_tokens'],seed=42,stop_token_ids=sorted(codec.eos),logprobs=16 if source else None,
            output_kind=RequestOutputKind.FINAL_ONLY,
            allowed_token_ids=codec.valid_ids if codec.valid_ids!=list(range(codec.support_size)) else None) for _ in requests]
        with progress.phase_scope('continuous_generation'):
            outputs=engine.generate([{'prompt_token_ids':r['prompt_token_ids']} for r in requests],params,
                                    steering=False,use_tqdm=progress.engine_tqdm)
        if len(outputs)!=len(requests):raise RuntimeError('Incomplete generation batch')
        progress.finished_call(len(requests))
        if source:
            positions=sum(len(response.outputs[0].token_ids) for response in outputs)
            require_artifact_space(output,positions*(16*8+4))
        files=[output/'job.json']
        with progress.phase_scope('save_complete_batch'):
            for request,response in zip(requests,outputs):
                if list(response.prompt_token_ids)!=request['prompt_token_ids']:raise ValueError('Generation changed prompt order/tokens')
                sample=response.outputs[0]
                row={**request,'token_ids':list(sample.token_ids),'text':sample.text,
                    'finish_reason':sample.finish_reason,'stop_reason':sample.stop_reason,
                    'sampled_logps':[float(lp[t].logprob) for t,lp in zip(sample.token_ids,sample.logprobs)] if source else [],
                    'model':job['model'],'steering':False,'job_hash':job_hash}
                if job.get('model_content_id'):row['model_content_id']=job['model_content_id']
                if source:
                    score=extract_topk(row['token_ids'],sample.logprobs,16,codec.support_size)
                    row.update(steering_mask=[False]*len(row['token_ids']),token_record_schema=2)
                    commit_topk_response(output,row,score);files.append(output/'topk'/f"{row['key']}.pt")
                else:atomic_json(output/f"{row['key']}.json",row)
                files.append(output/f"{row['key']}.json")
            stage.commit(files,requests=len(requests))
        progress.mark_committed(len(requests))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--job',required=True)
    args=parser.parse_args();generate(read_json(args.job))


if __name__=='__main__':main()
