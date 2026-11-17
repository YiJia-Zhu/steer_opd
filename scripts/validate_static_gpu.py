#!/usr/bin/env python3
"""Small EasySteer contract checks, with an optional separate HF comparison."""
import argparse

import torch

from steer_opd.static_masks import validate_native_score
from steer_opd.rollout_topk import extract_topk
from steer_opd.compass_job import load_codecs,model_path,project_path,MODEL_ROLES
from steer_opd.experiment_runtime import initialize_stage,StageCheckpoint,Progress
from steer_opd.math_data import prompt_ids
from steer_opd.resumable_core import atomic_json,digest,read_json

# Fixed BF16 acceptance tolerances, never widened on failure.
LIMITS={'mean_abs_logp':.05,'max_abs_logp':.2,'max_top16_tv':.03,'max_hidden_relative_l2':.03}


def compare_scores(reference,actual,*,strict=True):
    if len(reference)!=len(actual):raise ValueError('Validation score row count changed')
    errors=torch.cat([(a['sampled_logps']-b['sampled_logps']).abs() for a,b in zip(reference,actual)])
    max_tv=0.
    for a,b in zip(reference,actual):
        if a['ids'].shape != b['ids'].shape:
            raise ValueError('Validation Top16 positions changed')
        for record in (a,b):
            validate_native_score(record,len(record['sampled_logps']),int(record['ids'].max())+1)
        # TV over the union of the returned IDs; omitted tail is not compared.
        matches=a['ids'][:,:,None]==b['ids'][:,None,:]
        ap,bp=a['logps'].double().exp(),b['logps'].double().exp()
        overlap=torch.minimum(ap[:,:,None],bp[:,None,:]).masked_fill(~matches,0.).sum((1,2))
        tv=(ap.sum(-1)+bp.sum(-1)-2*overlap)/2
        max_tv=max(max_tv,float(tv.max()))
    result={'mean_abs_logp':float(errors.mean()),'max_abs_logp':float(errors.max()),'max_top16_tv':max_tv}
    if any(not torch.isfinite(torch.tensor(v)) for v in result.values()):raise ValueError('Nonfinite GPU comparison')
    within=all(v<=LIMITS[k] for k,v in result.items())
    if strict and not within:
        raise ValueError(f'GPU probability parity failed: {result}; limits={LIMITS}')
    return {**result,'within_parity_limits':within}


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
    if role not in MODEL_ROLES:raise ValueError('Only student and teacher validation is supported')
    if job.get('stage')!='static_gpu_acceptance' or job.get('optimizer_updates')!=0 or job.get('execution_hold'):
        raise ValueError('Expected an explicitly requested zero-update validation job')
    from steer_opd.stage_io import component_target
    output=(project_path(component_target(job,'acceptance')) if job.get('layout_version')==1
            else project_path(job['run_dir'])/'acceptance')/role/backend
    runtime=initialize_stage(job,output)
    store=StageCheckpoint(output,digest({'runtime':runtime,'binding':job['gpu_acceptance_binding'],'role':role,'backend':backend}))
    with Progress(output,'validate_'+role+'_'+backend,1,unit='check',show_bar=False) as progress:
        if store.load() is not None:progress.resume(1);return
        codecs=load_codecs(job);codec=codecs[role]
        use_hf=job.get('startup_hf_parity',False)
        validate_sv=job.get('stop_after','panel')=='panel'
        validate_role_sv=validate_sv and (role=='student' or job.get('include_teacher_ablation',False))
        capture_layers=job['layers'] if role=='student' else job.get('teacher_layers',[29])
        operator=job.get('operator','slerp')
        def scaled(v,strength):
            return {l:value*strength for l,value in v.items()} if operator=='additive' else v
        if backend=='hf':
            if not use_hf:raise ValueError('HF numerical comparison was not explicitly enabled')
            progress.phase='hf_reference'
            from steer_opd.compass_scoring import load_text_scorer
            from steer_opd.static_screen import score_batch
            model=load_text_scorer(model_path(job[role]),codec,codecs['teacher' if role=='student' else 'student'])
            rows=[{'id':str(i),'key':str(i),'prompt_token_ids':prompt_ids(codec.tokenizer,q,enable_thinking=False),
                   'token_ids':codec.tokenizer.encode(answer,add_special_tokens=False)+[min(codec.eos)]}
                  for i,(q,answer) in enumerate([('What is 2 + 3?','The answer is 5.'),('Compute 7 - 2.','5.')])]
            reference={'rows':rows,'baseline':score_batch(model,rows)}
            if validate_role_sv:
                states=hf_capture(model,rows,capture_layers)
                reference['captured']=states
            if validate_role_sv:
                vector={l:states[0][str(l)][1:].float().mean(0) for l in capture_layers}
                vectors=[vector,{l:-v.flip(0) for l,v in vector.items()}]
                gates=[[i%2==j for i in range(len(r['token_ids']))] for j,r in enumerate(rows)]
                reference.update(captured=states,vectors=vectors,gates=gates)
                for name,masks in [('static',gates),('all',[[True]*len(r['token_ids']) for r in rows])]:
                    reference[name]=score_batch(model,rows,vectors=[scaled(v,job['alpha']) for v in vectors],
                        gates=masks,operator=operator,alpha=job['alpha'] if operator=='slerp' else 1.)
                if 'linear_80_tail20' in job.get('strength_policies',[]):
                    reference['mixed_strength']=score_batch(model,rows,vectors=vectors,
                        gates=[[True]*len(r['token_ids']) for r in rows],alpha=[2*job['alpha'],0.])
            store.save_tensors({'reference':reference})
        else:
            from steer_opd.easysteer_static import create_engine,fixed_prefix_scores,build_spec
            if use_hf:
                reference=torch.load(output.parent/'hf/reference.pt',map_location='cpu',weights_only=True);rows=reference['rows']
            else:
                rows=[{'id':str(i),'key':str(i),'prompt_token_ids':prompt_ids(codec.tokenizer,q,enable_thinking=False),
                       'token_ids':codec.tokenizer.encode(answer,add_special_tokens=False)+[min(codec.eos)]}
                      for i,(q,answer) in enumerate([('What is 2 + 3?','The answer is 5.'),('Compute 7 - 2.','5.')])]
                reference={'rows':rows}
            engine=create_engine(model_path(job[role]),codec.support_size,max_model_len=256,
                tensor_parallel_size=job['teacher_gpus'] if role=='teacher' else 1,
                gpu_memory_utilization=job['gpu_memory_utilization'],
                progress=progress,batch_invariant=role=='student')
            progress.phase='numerical_parity'
            def score(specs):return fixed_prefix_scores(engine,rows,specs,include_sampled=True)
            def spec(row,mask,vector,*,alpha):
                return build_spec(row,mask,scaled(vector,alpha),operator=operator,alpha=alpha)
            baseline=score([False]*len(rows))
            for row,record in zip(rows,baseline):
                validate_native_score(record,len(row['token_ids']),codec.support_size)
                if not torch.isfinite(record['sampled_logps']).all() or (record['sampled_logps']>1e-6).any():
                    raise ValueError('Invalid observed-token native logprobs')
            report={'hf_parity':use_hf,'limits':LIMITS,
                'stop_after':job.get('stop_after','panel'),'sv_validation':validate_sv,
                'native_top16_support_and_observed_logps':'passed',
                'baseline':compare_scores(reference['baseline'] if use_hf else score([False]*len(rows)),baseline,
                                          strict=use_hf or role=='student')}
            if not use_hf and role!='student':
                report['repeat_drift']=report.pop('baseline')
                report['validation_scope']='raw Top16 cache; repeat drift recorded, not a student replay check'
                report['reuse_policy']='Teacher scored once and SHA-frozen for all student candidates'
            if validate_role_sv:
                import easysteer.capture as hs
                from vllm.capture import SelectSpec
                result=hs.capture(engine,[{'prompt_token_ids':r['prompt_token_ids']+r['token_ids']} for r in rows],
                    layers=capture_layers,dtype='bfloat16',steering=False,seed=42,
                    per_prompt_selects=[SelectSpec(prompt_positions=list(range(len(r['prompt_token_ids'])-1,
                        len(r['prompt_token_ids'])+len(r['token_ids'])))) for r in rows])
                if list(result.layer_ids)!=capture_layers or len(result.outputs)!=len(rows):
                    raise ValueError('Capture layers or sample count differ')
                if not use_hf:
                    reference['captured']=[{str(l):v.cpu() for l,v in result.sample(i).items()} for i in range(len(rows))]
                    if validate_role_sv:
                        vector={l:reference['captured'][0][str(l)][1:].float().mean(0) for l in capture_layers}
                        reference['vectors']=[vector,{l:-v.flip(0) for l,v in vector.items()}]
                        reference['gates']=[[i%2==j for i in range(len(r['token_ids']))] for j,r in enumerate(rows)]
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
                if use_hf:report['max_hidden_relative_l2']=hidden_error
                else:report['capture_positions_tokens_and_finite_states']='passed'
                report['capture_layers']=capture_layers
                report['operator']=operator
                report['strength']=job['alpha']
                report['additive_scaling']='raw_mean_difference_v1' if operator=='additive' else None
            if validate_role_sv:
                for name,masks in [('static',reference['gates']),('all',[[True]*len(r['token_ids']) for r in rows])]:
                    specs=[spec(r,g,v,alpha=job['alpha']) for r,g,v in zip(rows,masks,reference['vectors'])]
                    actual=score(specs)
                    expected=(reference[name] if use_hf else [fixed_prefix_scores(engine,[r],[s],include_sampled=True)[0]
                                for r,s in zip(rows,specs)])
                    report[name]=compare_scores(expected,actual,strict=use_hf or role=='student')
                    if name=='all' and job['alpha']>=.01 and all(torch.equal(a['sampled_logps'],b['sampled_logps']) for a,b in zip(baseline,actual)):
                        raise ValueError('Nonzero steering produced no change')
                if 'linear_80_tail20' in job.get('strength_policies',[]):
                    mixed=[spec(r,[True]*len(r['token_ids']),v,alpha=a)
                        for r,v,a in zip(rows,reference['vectors'],[2*job['alpha'],0.])]
                    actual=score(mixed)
                    expected=(reference['mixed_strength'] if use_hf else
                        [fixed_prefix_scores(engine,[r],[s],include_sampled=True)[0] for r,s in zip(rows,mixed)])
                    report['mixed_strength']=compare_scores(expected,actual)
                    torch.testing.assert_close(actual[1]['sampled_logps'],baseline[1]['sampled_logps'],atol=1e-5,rtol=1e-5)
                    report['mixed_strength_alpha']=[2*job['alpha'],0.]
                for name,specs in [('alpha0',[spec(r,[True]*len(r['token_ids']),v,alpha=0) for r,v in zip(rows,reference['vectors'])]),
                                   ('empty_mask',[spec(r,[False]*len(r['token_ids']),v,alpha=job['alpha']) for r,v in zip(rows,reference['vectors'])]),
                                   ('removed',[False]*len(rows))]:
                    unchanged=score(specs)
                    if role=='student':
                        for a,b in zip(baseline,unchanged):torch.testing.assert_close(a['sampled_logps'],b['sampled_logps'],atol=1e-5,rtol=1e-5)
                        report[name]='passed'
                    else:
                        report[name]=compare_scores(baseline,unchanged,strict=False)
                        report['teacher_steering_validation']='capture, finite native scores and explicit replay drift; batch-invariant unsupported'
            if role=='student':
                # Native source sampling and frozen-prefix replay use the same actual tokens.
                from vllm import SamplingParams
                outputs=engine.generate([{'prompt_token_ids':r['prompt_token_ids']} for r in rows],
                    SamplingParams(max_tokens=8,temperature=1,seed=42,logprobs=16,stop_token_ids=sorted(codec.eos)),
                    steering=False,use_tqdm=True)
                source=[];replay_rows=[]
                for row,out in zip(rows,outputs):
                    sample=out.outputs[0];record={**row,'token_ids':list(sample.token_ids)};replay_rows.append(record)
                    source.append(extract_topk(record['token_ids'],sample.logprobs,16,codec.support_size))
                replay=fixed_prefix_scores(engine,replay_rows,[False]*len(rows),include_sampled=True)
                report['source_replay']=compare_scores(source,replay)
            atomic_json(output/'report.json',report)
            store.commit([output/'report.json'])
        progress.finished_call(1);progress.mark_committed(1)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--job',required=True)
    p.add_argument('--role',choices=MODEL_ROLES,required=True)
    p.add_argument('--backend',choices=['hf','easysteer'],required=True)
    args=p.parse_args();run(read_json(args.job),args.role,args.backend)


if __name__=='__main__':main()
