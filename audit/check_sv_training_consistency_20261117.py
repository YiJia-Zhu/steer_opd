"""Read-only official-source, saved-bank and CPU numerical consistency audit."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Optional

import numpy as np
import torch

from steer_opd.direction_bank import bank_vectors
from steer_opd.experiment_runtime import source_revision
from steer_opd.extraction import pool_question
from steer_opd.operators import slerp
from steer_opd.resumable_core import ROOT, atomic_json
from steer_opd.static_masks import selector_for_predictions


def load_module(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def summary(values):
    x=torch.as_tensor(values).float()
    return {'mean':float(x.mean()),'min':float(x.min()),'p05':float(x.quantile(.05)),
            'median':float(x.median()),'p95':float(x.quantile(.95)),'max':float(x.max())}


def main():
    revision=source_revision()[0]
    official=ROOT/'.runtime/sv_training_audit_20261117'
    steering=load_module('_official_sv_audit_steering',official/'sv_training/steering.py')
    vectors=load_module('_official_sv_audit_vectors',official/'sv_training/vectors.py')
    report={'official_commit':'198fdbd917e9d7234f734aa8eb754faa408bd2e6',
            'source_revision':revision,'gpu_started':False,'production_code_changed':False,
            'source_files':{},'pinned_source_matches':{}}
    for name in ('acts','generate','train','vectors'):
        previous=ROOT/f'ref/implementation_sources/sv_{name}.py'
        fetched=official/f'sv_training/{name}.py'
        report['pinned_source_matches'][name]=previous.read_bytes()==fetched.read_bytes()
        assert report['pinned_source_matches'][name]
    suffix=Path('T_Qwen3.5-35B-A3B_S_Qwen3-4B/OpenThoughts3-1.2M/shared')
    run=ROOT/'outputs/step_3_overlap/ablation'/suffix/'n1000_slerp0.1_S26_T29_seed42_st_only_v1'
    bankroot=ROOT/'outputs/step_2_sv/ablation'/suffix/run.name/'reader_teacher'
    def load(path):
        report['source_files'][str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        return torch.load(path,map_location='cpu',weights_only=True)
    pooled=load(bankroot/'activations/pooled.pt')['pooled']
    global_bank=load(bankroot/'banks/S-T_global.pt')
    per_bank=load(bankroot/'banks/S-T_per_question.pt')
    ids=per_bank['valid_ids'];assert ids==global_bank['valid_ids']
    target=torch.stack([pooled[q]['layers']['29']['pools'][0] for q in ids])
    own=torch.stack([pooled[q]['layers']['29']['pools'][1] for q in ids])
    delta=target-own;unit=delta/delta.norm(dim=-1,keepdim=True)
    official_mean=vectors.mean_vector_from_acts({'gt_acts':target,'pred_acts':own},29)
    assert torch.equal(official_mean,global_bank['layers']['29']['direction'])
    for i,qid in enumerate(ids):
        assert torch.equal(delta[i],per_bank['per_question'][qid]['29']['direction'])
        actual=bank_vectors(per_bank,qid,operator='slerp',alpha=.1,layers=[29])[0][29]
        assert torch.equal(actual,steering.Injection(delta[i],29,alpha=.1,op='slerp')._table)
    mean_unit=official_mean/official_mean.norm()
    alignment=(unit*mean_unit).sum(-1)
    rounded=target.bfloat16().float()-own.bfloat16().float()
    rounded=rounded/rounded.norm(dim=-1,keepdim=True)
    report['bank_checks']={'questions':len(ids),'reader':'teacher','direction':'S-T',
        'all_raw_differences_exact':True,'global_matches_official_mean_exact':True,
        'all_unit_directions_match_official_exact':True,
        'zero_directions':int((delta.norm(dim=-1)<1e-8).sum()),
        'per_question_to_global_cosine':summary(alignment),
        'per_question_raw_norm':summary(delta.norm(dim=-1)),
        'global_raw_norm':float(official_mean.norm()),
        'fp32_vs_bf16_rounded_pool_direction_cosine':summary((unit*rounded).sum(-1))}
    bindings=[];all_rows=[]
    for split in ('select','confirm'):
        rows=json.loads((run/f'{split}_rows.json').read_text());all_rows+=rows
        locations=np.linspace(0,len(rows)-1,16,dtype=int)
        packet=json.loads((run/split/'teacher/summaries.json').read_text())
        for scheme,bank in [('scheme_05',global_bank),('scheme_06',per_bank)]:
            condition=packet[scheme]['condition']
            for index in locations:
                row=rows[int(index)];path=run/split/'teacher'/scheme/(row['key']+'.pt')
                saved=load(path)
                _,expected=bank_vectors(bank,row['id'],operator=condition['operator'],
                    alpha=condition['alpha'],layers=condition['layers'])
                assert saved['vector_id']==expected and saved['condition']==condition
                bindings.append({'split':split,'scheme':scheme,'qid':row['id'],'key':row['key'],'matches':True})
    assert len(all_rows)==len(ids)==len({row['id'] for row in all_rows})
    assert {row['id'] for row in all_rows}==set(ids)
    report['bindings']={'all_1000_question_ids_covered':True,'saved_records_checked':bindings}
    report['geometry_and_outcome']={}
    idpos={qid:i for i,qid in enumerate(ids)}
    for split in ('select','confirm'):
        packet=json.loads((run/split/'teacher/summaries.json').read_text())
        g=packet['scheme_05']['metrics']['overlap'];p=packet['scheme_06']['metrics']['overlap']
        keys=sorted(p['per_question_delta_pp'])
        advantage=np.array([g['per_question_delta_pp'][q]-p['per_question_delta_pp'][q] for q in keys])
        cos=np.array([float(alignment[idpos[q]]) for q in keys])
        report['geometry_and_outcome'][split]={
            'global_delta_pp':g['question_weighted_delta_pp'],'per_question_delta_pp':p['question_weighted_delta_pp'],
            'per_question_improved_questions':p['improved_questions'],'questions':p['questions'],
            'corr_alignment_vs_global_minus_per_question_gain':float(np.corrcoef(cos,advantage)[0,1])}
    comparisons=[]
    generator=torch.Generator().manual_seed(42)
    for dtype in (torch.float32,torch.bfloat16):
        h=torch.randn((128,2048),generator=generator).to(dtype)
        raw=torch.randn((128,2048),generator=generator)
        v=(raw/raw.norm(dim=-1,keepdim=True)).to(dtype)
        for alpha in (.05,.1,.2):
            a=steering.slerp_rotate(h,v,alpha);b=slerp(h,v,alpha)
            error=a.float()-b.float()
            comparisons.append({'dtype':str(dtype),'alpha':alpha,'max_abs_error':float(error.abs().max()),
                'relative_l2_error':float(error.norm()/a.float().norm()),
                'bitwise_equal':bool(torch.equal(a,b)),'both_finite':bool(torch.isfinite(a).all() and torch.isfinite(b).all())})
    report['slerp_numerics']=comparisons
    row={'prompt_token_ids':[1,2,3,4,5],'token_ids':[6,7,8,9]}
    labels=torch.tensor([[-100]*5+[6,7,8,9]])
    official_positions=(labels[0]!=-100).nonzero().flatten().tolist()
    ours=selector_for_predictions(row,[True]*4,mode='fixed_prefix')['prompt_positions']
    report['position_protocol']={'official_labels_mask_positions':official_positions,
        'our_prediction_state_positions':ours,'same':official_positions==ours,
        'our_minus_official':[a-b for a,b in zip(ours,official_positions)]}
    # A tiny random CPU Qwen model tests pooling conventions without loading weights.
    from transformers import AutoTokenizer,Qwen3Config,Qwen3ForCausalLM
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'assets/models/Qwen3-4B',local_files_only=True)
    node=next(n for n in ast.parse((official/'sv_training/acts.py').read_text()).body
              if isinstance(n,ast.FunctionDef) and n.name=='mean_hidden_all_layers')
    namespace={'torch':torch,'Optional':Optional}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(official/'sv_training/acts.py'),'exec'),namespace)
    prompt='Problem: 1+1=?\nAnswer: ';answer='We add one and one to get two. Therefore, 2.'
    pooling=[]
    for dtype in (torch.float32,torch.bfloat16):
        torch.manual_seed(42)
        cfg=Qwen3Config(vocab_size=len(tokenizer.get_vocab()),hidden_size=16,intermediate_size=32,
            num_hidden_layers=3,num_attention_heads=2,num_key_value_heads=1,head_dim=8)
        cfg._attn_implementation='eager'
        model=Qwen3ForCausalLM(cfg).to(dtype).eval()
        expected=namespace['mean_hidden_all_layers'](model,tokenizer,prompt,answer,2048)
        actual=pool_question(model,tokenizer,{'reader_prompt_text':prompt},[{'text':answer}],list(range(3)),allow_zero=True)
        for layer in range(3):
            got=actual['layers'][str(layer)]['pools'][0];wanted=expected[layer]
            pooling.append({'dtype':str(dtype),'layer':layer,'last_layer':layer==2,
                'max_abs_error':float((got-wanted).abs().max()),
                'relative_l2_error':float((got-wanted).norm()/wanted.norm()),
                'equal_after_bf16_round':bool(torch.equal(got.bfloat16().float(),wanted)),
                'bitwise_equal':bool(torch.equal(got,wanted))})
    report['pooling_numerics']=pooling
    assert not torch.cuda.is_initialized()
    assert source_revision()[0]==revision
    atomic_json(ROOT/'audit/sv_training_consistency_20261117.json',report)
    print(json.dumps({k:report[k] for k in ('bank_checks','geometry_and_outcome','slerp_numerics','position_protocol','pooling_numerics')},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
