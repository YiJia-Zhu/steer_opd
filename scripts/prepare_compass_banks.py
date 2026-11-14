#!/usr/bin/env python3
"""Extract the shared S0 activation pool once and commit all four banks together."""
import argparse
import torch
from transformers import AutoTokenizer

from steer_opd.compass_banks import make_common_banks
from steer_opd.compass_job import project_path, model_path
from steer_opd.compass_alignment import ByteLevelCodec
from steer_opd.compass_scoring import load_text_scorer
from steer_opd.direction_bank import file_sha256
from steer_opd.extraction import pool_question, usable_text
from steer_opd.execution_policy import require_execution_allowed
from steer_opd.resumable_core import atomic_json, atomic_torch, digest, read_json
from steer_opd.experiment_runtime import initialize_stage, StageCheckpoint, Progress


def build(config):
    require_execution_allowed(config)
    output=project_path(config['bank_dir']);runtime=initialize_stage(config,output)
    questions=read_json(project_path(config['questions']));roles=('teacher','reference','student')
    content_ids={}
    if config.get('bind_checkpoint_content'):
        from steer_opd.checkpoint_identity import checkpoint_content_id
        content_ids={r:checkpoint_content_id(model_path(config[r]),receipt_dir=output/'provenance/checkpoints') for r in roles}
    inputs=[];identities={}
    for question in questions:
        records=[read_json(project_path(config['reference_dirs'][role])/f"{question['id']}.json") for role in roles]
        if any(r['id']!=question['id'] or r.get('model')!=config[role] for role,r in zip(roles,records)):
            raise ValueError('Reference text question/model differs from configured role')
        if content_ids and any(r.get('model_content_id')!=content_ids[role] for role,r in zip(roles,records)):
            raise ValueError('Reference checkpoint contents changed')
        identity={'checkpoint':config['student'],'step':0,'layers':config['layers'],
            'question_hash':digest(question),'roles':list(roles),'text_record_hashes':[digest(r) for r in records],
            'boundary':'block_output_before_final_norm','backend':config.get('extraction_backend','hf_batched_static'),
            'runtime_binding':runtime,'checkpoint_content_ids':content_ids}
        identities[question['id']]=identity;inputs.append((question,records,identity))
    stage=StageCheckpoint(output,digest({'config':config,'runtime':runtime,'inputs':identities}))
    with Progress(output/'bank_progress','direction_banks',len(questions),unit='question') as progress:
        if stage.load() is not None:progress.resume(len(questions));return
        pooled={q['id']:{'usable':False,'reason':'common_text_quality_filter','identity':identity}
                for q,records,identity in inputs if not all(usable_text(r) for r in records)}
        valid_inputs=[item for item in inputs if item[0]['id'] not in pooled]
        tokenizer=AutoTokenizer.from_pretrained(model_path(config['student']),local_files_only=True)
        backend=config.get('extraction_backend','hf_batched_static')
        if backend=='easysteer':
            from steer_opd.easysteer_extraction import extract_all
            extract_all(config,tokenizer,valid_inputs,pooled)
        elif backend=='hf_batched_static':
            # Explicit numerical reference; the production configuration selects EasySteer.
            model=load_text_scorer(model_path(config['student']),ByteLevelCodec(tokenizer))
            model.model.layers=torch.nn.ModuleList(list(model.model.layers[:max(config['layers'])+1]))
            for question,records,identity in valid_inputs:
                artifacts={}
                def save(index,payload):
                    path=output/'token_activations'/question['id']/f'{roles[index]}.pt'
                    atomic_torch(path,{**payload,'identity':identity,'role':roles[index]})
                    artifacts[roles[index]]={'path':str(path.relative_to(project_path('.'))),'sha256':file_sha256(path)}
                pooled[question['id']]={**pool_question(model,tokenizer,question,records,config['layers'],save,
                    cache_dtype=torch.bfloat16,allow_zero=True),'identity':identity,'token_activations':artifacts}
        else:raise ValueError('Unknown extraction backend; no automatic fallback')
        valid=[q for q in questions if pooled[q['id']]['usable']]
        if len(valid)<config.get('minimum_pairs',8):raise ValueError('Too few common valid reference triplets')
        identity={'checkpoint':config['student'],'step':0,'reference_pool_hash':digest(identities)}
        if content_ids:identity['checkpoint_content_id']=content_ids['student']
        with progress.phase_scope('save_complete_extraction_and_four_banks'):
            # Freeze question order independently of capture completion order.
            pooled={q['id']:pooled[q['id']] for q in questions}
            banks,scales=make_common_banks(pooled,config['layers'],identity,alpha=config['alpha'])
            files=[];hashes={}
            for qid,value in pooled.items():
                path=output/'pooled'/f'{qid}.pt';atomic_torch(path,value);files.append(path)
                files.extend(project_path(a['path']) for a in value.get('token_activations',{}).values())
            for name,bank in banks.items():
                path=output/f'{name}.pt';atomic_torch(path,bank);files.append(path);hashes[name]=file_sha256(path)
            atomic_torch(output/'common_scales.pt',scales);atomic_json(output/'questions.json',valid)
            atomic_json(output/'manifest.json',{'valid_ids':[q['id'] for q in valid],'bank_sha256':hashes,
                'rejected':{qid:v.get('reason','unusable_activation') for qid,v in pooled.items() if not v['usable']},
                'identity_hash':digest(identities),'parameter_updates':0,'layers':config['layers'],'alpha':config['alpha']})
            files.extend(output/name for name in ('common_scales.pt','questions.json','manifest.json'))
            stage.commit(files,questions=len(questions),banks=4)
        progress.finished_call(len(questions));progress.mark_committed(len(questions))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    args=parser.parse_args();build(read_json(args.config))


if __name__=='__main__':main()
