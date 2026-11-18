#!/usr/bin/env python3
"""One whole-call capture of paired answers in the selected frozen reader space."""
import argparse

from transformers import AutoTokenizer

from steer_opd.checkpoint_identity import checkpoint_content_id
from steer_opd.compass_job import project_path, model_path
from steer_opd.direction_bank import file_sha256
from steer_opd.execution_policy import require_execution_allowed
from steer_opd.experiment_runtime import initialize_stage, StageCheckpoint, Progress, compute_required
from steer_opd.extraction import encode_answer
from steer_opd.math_data import prompt_text
from steer_opd.stage_io import artifact_dir
from steer_opd.method_config import DIRECTION_VERSION, DIRECTION_VERSIONS, DIRECTION_ROLES, reader_direction, reader_layers
from steer_opd.resumable_core import atomic_json, atomic_torch, read_json, digest


def capture(cfg, reader_role="student"):
    require_execution_allowed(cfg)
    if cfg.get('direction')!='T-S' or cfg.get('direction_version')!=DIRECTION_VERSION:
        raise ValueError('Expected the current T-S capture configuration')
    direction=reader_direction(reader_role)
    if reader_role=='teacher' and not cfg.get('include_teacher_ablation'):
        raise ValueError('Teacher-space capture requires explicit teacher ablation selection')
    roles=DIRECTION_ROLES[direction]
    run=project_path(cfg['run_dir']);output=project_path(artifact_dir(cfg,'capture_'+reader_role))
    content={role:checkpoint_content_id(model_path(cfg[role]),receipt_dir=output/'provenance/checkpoints')
             for role in ('teacher','student')}
    tokenizer=AutoTokenizer.from_pretrained(model_path(cfg['tokenizer_paths'][reader_role]),local_files_only=True)
    questions=read_json(project_path(cfg['data_dir'])/'questions.json');records=[];encodings={}
    for q in questions:
        texts={'teacher':read_json(project_path(artifact_dir(cfg,'teacher'))/f"{q['id']}.json"),
               'student':read_json(project_path(artifact_dir(cfg,'student'))/f"{q['id']}_0.json")}
        pair=[texts[role] for role in roles]
        for role,record in zip(roles,pair):
            if (record['id']!=q['id'] or record['model']!=cfg[role]
                    or record.get('model_content_id')!=content[role]):
                raise ValueError('Teacher/self text question or checkpoint contents changed')
        item={**q,'reader_prompt_text':prompt_text(tokenizer,q['question'],cfg['enable_thinking'])}
        encoded=[encode_answer(tokenizer,item['reader_prompt_text'],r.get('text'),cfg['sv_max_context_len']) for r in pair]
        encodings[q['id']]=encoded
        records.append((item,pair,all(e['response_ids'] for e in encoded)))
    valid_ids={q['id'] for q,_,valid in records if valid}
    if cfg.get('data_policy')=='all_train':
        quality_scope={'data_policy':'all_train','training_usable_questions':len(valid_ids),
                       'ready_for_capture':bool(valid_ids)}
    else:
        split=read_json(project_path(cfg['data_dir'])/'split.json')
        counts={name:len(valid_ids.intersection(split[name+'_ids'])) for name in ('select','confirm')}
        quality_scope={'split_usable_counts':counts,'minimum_split_questions':cfg['minimum_split_questions'],
            'ready_for_capture':all(n>=cfg['minimum_split_questions'] for n in counts.values())}
    quality={'input_questions':len(records),'paired_usable_questions':len(valid_ids),
        **quality_scope,
        'roles':{role:{'length_truncated':sum(pair[i]['finish_reason']=='length' for _,pair,_ in records),
            'empty':sum(not pair[i].get('text','').strip() for _,pair,_ in records),
            'context_truncated':sum(encodings[q['id']][i]['context_truncated'] for q,_,_ in records),
            'usable':sum(bool(encodings[q['id']][i]['response_ids']) for q,_,_ in records)}
            for i,role in enumerate(roles)},
        'rejected_question_ids':[q['id'] for q,_,valid in records if not valid],
        'filter':'nonempty_answer_tokens_after_context_truncation',
        'sv_max_context_len':cfg['sv_max_context_len'],
        'finish_reason_correctness_repetition_filter':False,
        'pooling':'sv_training_all_retained_answer_tokens',
        'input_manifest_sha256':{role:file_sha256(project_path(artifact_dir(cfg,role))/'complete.json')
            for role in ('teacher','student')}}
    if (output/'complete.json').exists():
        if read_json(output/'text_quality.json')!=quality:
            raise ValueError('Committed capture text quality inputs changed')
    else:atomic_json(output/'text_quality.json',quality)
    if not quality['ready_for_capture']:
        detail=(f"training_usable={len(valid_ids)}; need at least one pair" if cfg.get('data_policy')=='all_train'
                else f"select={counts['select']}, confirm={counts['confirm']}; need at least {cfg['minimum_split_questions']} per split")
        raise ValueError(f"Insufficient readable teacher/self answer pairs before GPU capture: {detail}. See {output/'text_quality.json'}")
    runtime=initialize_stage(cfg,output)
    layers=reader_layers(cfg,reader_role)
    identity={'checkpoint':cfg[reader_role],'checkpoint_content_id':content[reader_role],'source_checkpoint_contents':content,
        'layer':layers[0],'reader_role':reader_role,'direction':direction,'pool_roles':list(roles),'tokenizer_sha256':digest(tokenizer.get_vocab()),
        'template_sha256':digest(tokenizer.chat_template),'enable_thinking':cfg['enable_thinking'],
        'pooling':'sv_training_all_retained_answer_tokens','sv_max_context_len':cfg['sv_max_context_len'],
        'direction_version':DIRECTION_VERSIONS[direction],
        'runtime_binding':runtime,'texts_hash':digest([(q['id'],q['question'],rs,valid) for q,rs,valid in records])}
    stage=StageCheckpoint(output,digest(identity))
    with Progress(output/'progress',reader_role+'_paired_answer_capture',len(records),unit='question',show_bar=False) as progress:
        if stage.load() is not None:progress.resume(len(records));return
        compute_required()
        pooled={q['id']:{'usable':False,'reason':'no_answer_content'} for q,rs,valid in records if not valid}
        inputs=[]
        for q,rs,valid in records:
            if valid:
                inputs.append((q,rs,{**identity,'question_hash':digest(q),'text_record_hashes':[digest(r) for r in rs]}))
        if cfg['extraction_backend']=='easysteer':
            from steer_opd.easysteer_extraction import extract_all
            extract_all({**cfg,'layers':layers,'reader_role':reader_role,'bank_dir':str(output.relative_to(project_path('.')))},tokenizer,inputs,pooled)
        elif cfg['extraction_backend']=='hf_batched_static':
            from steer_opd.compass_alignment import ByteLevelCodec
            from steer_opd.compass_scoring import load_text_scorer
            from steer_opd.extraction import pool_question
            model=load_text_scorer(model_path(cfg[reader_role]),ByteLevelCodec(tokenizer))
            for q,rs,item_identity in inputs:
                pooled[q['id']]={**pool_question(model,tokenizer,q,rs,layers,allow_zero=True,
                    max_context_len=cfg['sv_max_context_len']),
                                  'identity':item_identity}
        else:raise ValueError('Unknown explicit extraction backend')
        path=output/'pooled.pt';atomic_torch(path,{'identity':identity,'pooled':pooled});files=[path]
        files.append(output/'text_quality.json')
        if cfg['extraction_backend']=='easysteer':files.append(output/'capture_budget.json')
        stage.commit(files,questions=len(records),hidden_state_storage='pooled_only',
                     retention_policy=cfg.get('retention_policy'))
        progress.finished_call(len(records));progress.mark_committed(len(records))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    parser.add_argument('--reader',choices=['student','teacher'],default='student')
    args=parser.parse_args();capture(read_json(args.config),args.reader)
