#!/usr/bin/env python3
"""Small, zero-update GPU acceptance; separate HF/engine processes release memory."""
import argparse

import torch

from steer_opd.byte_probabilities import projection_spec,validate_byte_probabilities
from steer_opd.compass_job import load_codecs,model_path,project_path
from steer_opd.experiment_runtime import initialize_stage,StageCheckpoint,Progress
from steer_opd.math_data import prompt_ids
from steer_opd.resumable_core import atomic_json,atomic_torch,digest,read_json

# Fixed BF16 acceptance tolerances, never widened on failure.
LIMITS={'mean_abs_logp':.05,'max_abs_logp':.2,'max_byte_tv':.03,'max_hidden_relative_l2':.03}


def compare_scores(reference,actual):
    if len(reference)!=len(actual):raise ValueError('Validation score row count changed')
    for a,b in zip(reference,actual):
        for value in (a,b):validate_byte_probabilities(value['byte_probs'],len(value['sampled_logps']))
    errors=torch.cat([(a['sampled_logps']-b['sampled_logps']).abs() for a,b in zip(reference,actual)])
    tv=float(torch.stack([(a['byte_probs']-b['byte_probs']).abs().sum(-1).max()/2 for a,b in zip(reference,actual)]).max())
    result={'mean_abs_logp':float(errors.mean()),'max_abs_logp':float(errors.max()),'max_byte_tv':tv}
    if any(not torch.isfinite(torch.tensor(v)) or v>LIMITS[k] for k,v in result.items()):
        raise ValueError(f'GPU probability parity failed: {result}; limits={LIMITS}')
    return result


def hf_capture(model,rows,layers):
    device=model._input_device;lengths=[len(r['prompt_token_ids'])+len(r['token_ids']) for r in rows]
    tokens=torch.zeros((len(rows),max(lengths)),dtype=torch.long,device=device);attention=torch.zeros_like(tokens)
    for i,row in enumerate(rows):
        tokens[i,:lengths[i]]=torch.tensor(row['prompt_token_ids']+row['token_ids'],device=device);attention[i,:lengths[i]]=1
    captured={};hooks=[]
    for layer in layers:
        def save(module,args,value,layer=layer):
            captured[layer]=(value[0] if isinstance(value,tuple) else value).detach().cpu().clone()
        hooks.append(model.model.layers[layer].register_forward_hook(save))
    try:
        with torch.inference_mode():model.model(input_ids=tokens,attention_mask=attention,use_cache=False)
    finally:
        for hook in hooks:hook.remove()
    return [{str(l):captured[l][i,len(r['prompt_token_ids'])-1:lengths[i]] for l in layers} for i,r in enumerate(rows)]


def run(job,role,backend):
    if job.get('stage')!='static_gpu_acceptance' or job.get('optimizer_updates')!=0 or job.get('execution_hold'):
        raise ValueError('Expected an explicitly requested zero-update validation job')
    output=project_path(job['run_dir'])/'acceptance'/role/backend
    runtime=initialize_stage(job,output)
    store=StageCheckpoint(output,digest({'runtime':runtime,'binding':job['gpu_acceptance_binding'],'role':role,'backend':backend}))
    with Progress(output,'validate_'+role+'_'+backend,1,unit='check',show_bar=False) as progress:
        if store.load() is not None:progress.resume(1);return
        codecs=load_codecs(job);codec=codecs[role]
        if backend=='hf':
            progress.phase='hf_reference'
            from steer_opd.compass_scoring import load_text_scorer
            from steer_opd.static_screen import score_batch
            model=load_text_scorer(model_path(job[role]),codec,codecs['teacher' if role=='student' else 'student'])
            rows=[{'id':str(i),'key':str(i),'prompt_token_ids':prompt_ids(codec.tokenizer,q,enable_thinking=False),
                   'token_ids':codec.tokenizer.encode(answer,add_special_tokens=False)+[min(codec.eos)]}
                  for i,(q,answer) in enumerate([('What is 2 + 3?','The answer is 5.'),('Compute 7 - 2.','5.')])]
            reference={'rows':rows,'baseline':score_batch(model,rows)}
            if role=='student':
                states=hf_capture(model,rows,job['layers'])
                vector={l:states[0][str(l)][1:].float().mean(0) for l in job['layers']}
                vectors=[vector,{l:-v.flip(0) for l,v in vector.items()}]
                gates=[[i%2==j for i in range(len(r['token_ids']))] for j,r in enumerate(rows)]
                reference.update(captured=states,vectors=vectors,gates=gates)
                for name,masks in [('static',gates),('all',[[True]*len(r['token_ids']) for r in rows])]:
                    reference[name]=score_batch(model,rows,vectors=vectors,gates=masks,alpha=job['alpha'])
            store.save_tensors({'reference':reference})
        else:
            from steer_opd.easysteer_static import create_engine,fixed_prefix_scores,build_spec
            from steer_opd.engine_projection import select_prediction_statistics
            ref_path=output.parent/'hf/reference.pt'
            reference=torch.load(ref_path,map_location='cpu',weights_only=True);rows=reference['rows']
            projection=output/'projection.pt';atomic_torch(projection,projection_spec(codec,codecs['teacher' if role=='student' else 'student']))
            engine=create_engine(model_path(job[role]),codec.support_size,max_model_len=256,
                tensor_parallel_size=job['teacher_gpus'] if role=='teacher' else 1,
                gpu_memory_utilization=job['gpu_memory_utilization'],projection_path=projection,
                capture_sampled=role=='student',progress=progress)
            progress.phase='numerical_parity'
            def score(specs):return fixed_prefix_scores(engine,rows,specs,include_sampled=True,collect_byte_probs=True)
            baseline=score([False]*len(rows));report={'baseline':compare_scores(reference['baseline'],baseline),'limits':LIMITS}
            if role=='student':
                import easysteer.capture as hs
                from vllm.capture import SelectSpec
                result=hs.capture(engine,[{'prompt_token_ids':r['prompt_token_ids']+r['token_ids']} for r in rows],
                    layers=job['layers'],dtype='bfloat16',steering=False,seed=42,
                    per_prompt_selects=[SelectSpec(prompt_positions=list(range(len(r['prompt_token_ids'])-1,
                        len(r['prompt_token_ids'])+len(r['token_ids'])))) for r in rows])
                if list(result.layer_ids)!=job['layers'] or len(result.outputs)!=len(rows):
                    raise ValueError('Capture layers or sample count differ')
                hidden_error=0.
                for i,row in enumerate(rows):
                    expected=list(range(len(row['prompt_token_ids'])-1,len(row['prompt_token_ids'])+len(row['token_ids'])))
                    tokens=row['prompt_token_ids']+row['token_ids']
                    if result.sample_positions(i)!=expected or result.sample_token_ids(i)!=[tokens[p] for p in expected]:
                        raise ValueError('Capture position/token labels differ')
                    for layer,values in result.sample(i).items():
                        other=reference['captured'][i][str(layer)].float()
                        delta=(values.float()-other).norm(dim=-1)/other.norm(dim=-1).clamp_min(1e-8)
                        if not torch.isfinite(delta).all():raise ValueError('Nonfinite captured states')
                        hidden_error=max(hidden_error,float(delta.max()))
                if not torch.isfinite(torch.tensor(hidden_error)) or hidden_error>LIMITS['max_hidden_relative_l2']:
                    raise ValueError(f'Capture parity failed: relative L2={hidden_error}')
                report['max_hidden_relative_l2']=hidden_error
                for name,masks in [('static',reference['gates']),('all',[[True]*len(r['token_ids']) for r in rows])]:
                    actual=score([build_spec(r,g,v,alpha=job['alpha']) for r,g,v in zip(rows,masks,reference['vectors'])])
                    report[name]=compare_scores(reference[name],actual)
                    if name=='all' and job['alpha']>=.01 and all(torch.equal(a['sampled_logps'],b['sampled_logps']) for a,b in zip(baseline,actual)):
                        raise ValueError('Nonzero steering produced no change')
                for name,specs in [('alpha0',[build_spec(r,[True]*len(r['token_ids']),v,alpha=0) for r,v in zip(rows,reference['vectors'])]),
                                   ('empty_mask',[build_spec(r,[False]*len(r['token_ids']),v,alpha=job['alpha']) for r,v in zip(rows,reference['vectors'])]),
                                   ('removed',[False]*len(rows))]:
                    unchanged=score(specs)
                    for a,b in zip(baseline,unchanged):torch.testing.assert_close(a['sampled_logps'],b['sampled_logps'],atol=1e-5,rtol=1e-5)
                    report[name]='passed'
                # Native source sampling and frozen-prefix replay use the same actual tokens.
                from vllm import SamplingParams
                outputs=engine.generate([{'prompt_token_ids':r['prompt_token_ids']} for r in rows],
                    SamplingParams(max_tokens=8,temperature=1,seed=42,logprobs=16,stop_token_ids=sorted(codec.eos)),
                    steering=False,use_tqdm=True)
                source=[];replay_rows=[]
                for row,out in zip(rows,outputs):
                    sample=out.outputs[0];record={**row,'token_ids':list(sample.token_ids)};replay_rows.append(record)
                    stats=select_prediction_statistics(engine.collective_rpc('project_pop_byte_statistics',args=(out.request_id,)),record)
                    source.append({**stats,'sampled_logps':torch.tensor([float(step[t].logprob) for t,step in zip(sample.token_ids,sample.logprobs)])})
                replay=fixed_prefix_scores(engine,replay_rows,[False]*len(rows),include_sampled=True,collect_byte_probs=True)
                report['source_replay']=compare_scores(source,replay)
            atomic_json(output/'report.json',report)
            store.commit([projection,output/'report.json'])
        progress.finished_call(1);progress.mark_committed(1)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--job',required=True)
    p.add_argument('--role',choices=['student','reference','teacher'],required=True)
    p.add_argument('--backend',choices=['hf','easysteer'],required=True)
    args=p.parse_args();run(read_json(args.job),args.role,args.backend)


if __name__=='__main__':main()
