#!/usr/bin/env python3
"""Build reader-specific global/per-question banks from committed pooled states."""
import argparse

import torch

from steer_opd.checkpoint_identity import checkpoint_content_id
from steer_opd.compass_banks import make_common_banks
from steer_opd.stage_io import artifact_dir
from steer_opd.compass_job import project_path, model_path
from steer_opd.direction_bank import file_sha256
from steer_opd.execution_policy import require_execution_allowed
from steer_opd.experiment_runtime import initialize_stage, StageCheckpoint, Progress, compute_required
from steer_opd.method_config import DIRECTION_VERSION, DIRECTION_VERSIONS, reader_direction, reader_layers
from steer_opd.resumable_core import atomic_json, atomic_torch, digest, read_json


def build(config, reader_role="student"):
    require_execution_allowed(config)
    if config.get('direction')!='T-S' or config.get('direction_version')!=DIRECTION_VERSION:
        raise ValueError('Expected the current T-S bank configuration')
    direction=reader_direction(reader_role);layers=reader_layers(config,reader_role)
    if reader_role=='teacher' and not config.get('include_teacher_ablation'):
        raise ValueError('Teacher banks require explicit teacher ablation selection')
    run=project_path(config['run_dir']);output=project_path(artifact_dir(config,'bank_'+reader_role))
    runtime=initialize_stage(config,output)
    questions=read_json(project_path(config['data_dir'])/'questions.json')
    capture_path=project_path(artifact_dir(config,'capture_'+reader_role))/'pooled.pt';capture=torch.load(capture_path,weights_only=True,map_location='cpu')
    meta=capture['identity']
    if StageCheckpoint(capture_path.parent,digest(meta)).load() is None:
        raise ValueError('Student activation pool is not committed')
    if (meta['checkpoint']!=config[reader_role] or meta['layer']!=layers[0]
            or meta['direction_version']!=DIRECTION_VERSIONS[direction]
            or meta['checkpoint_content_id']!=checkpoint_content_id(model_path(config[reader_role]))):
        raise ValueError('Student activation checkpoint, layer or direction protocol changed')
    if set(capture['pooled'])!={q['id'] for q in questions}:
        raise ValueError('Activation questions differ from the frozen campaign pool')
    paired_pool=None
    if config.get('include_teacher_ablation'):
        other=project_path(artifact_dir(config,'capture_'+('teacher' if reader_role=='student' else 'student')))/'pooled.pt'
        paired_pool=torch.load(other,weights_only=True,map_location='cpu')
        if StageCheckpoint(other.parent,digest(paired_pool['identity'])).load() is None:
            raise ValueError('The paired reader activation pool is not committed')
        other_role='teacher' if reader_role=='student' else 'student'
        other_meta=paired_pool['identity']
        if (other_meta['checkpoint']!=config[other_role]
                or other_meta['layer']!=reader_layers(config,other_role)[0]
                or other_meta['direction_version']!=DIRECTION_VERSIONS[reader_direction(other_role)]
                or other_meta['checkpoint_content_id']!=checkpoint_content_id(model_path(config[other_role]))):
            raise ValueError('The paired reader checkpoint, layer or direction protocol changed')
        if set(paired_pool['pooled'])!=set(capture['pooled']):
            raise ValueError('Reader pools must contain the same frozen question IDs')
    identity={'checkpoint':config[reader_role],'step':0,'checkpoint_content_id':meta['checkpoint_content_id'],
              'activation_pool_hash':file_sha256(capture_path),'direction_version':DIRECTION_VERSIONS[direction]}
    if config.get('data_policy')=='all_train':
        identity.update(data_policy='all_train',question_source_sha256=file_sha256(project_path(config['question_source'])),
                        training_ids_sha256=digest([q['id'] for q in questions]))
    if paired_pool is not None:identity['paired_activation_pool_hash']=file_sha256(other)
    stage=StageCheckpoint(output,digest({'config':config,'runtime':runtime,'identity':identity}))
    with Progress(output/'bank_progress','direction_banks',len(questions),unit='question',show_bar=False) as progress:
        if stage.load() is not None:progress.resume(len(questions));return
        compute_required()
        pooled={q['id']:capture['pooled'][q['id']] for q in questions}
        if paired_pool is not None:
            pooled={qid:record if paired_pool['pooled'][qid]['usable'] else
                    {'usable':False,'reason':'unusable_in_paired_reader'} for qid,record in pooled.items()}
        banks,scales=make_common_banks(pooled,layers,identity,alpha=config['alpha'],direction=direction)
        valid_ids=next(iter(banks.values()))['valid_ids']
        minimum=1 if config.get('data_policy')=='all_train' else 2*config['minimum_split_questions']
        if len(valid_ids)<minimum:raise ValueError('Too few valid teacher/self activation pairs')
        files=[];hashes={}
        for name,bank in banks.items():
            path=output/f'{name}.pt';atomic_torch(path,bank);files.append(path);hashes[name]=file_sha256(path)
        atomic_torch(output/'common_scales.pt',scales)
        atomic_json(output/'questions.json',[q for q in questions if q['id'] in valid_ids])
        atomic_json(output/'manifest.json',{'valid_ids':valid_ids,'bank_sha256':hashes,'identity':identity,
            'parameter_updates':0,'layers':layers,'reader_role':reader_role,'direction':direction,'alpha':config['alpha'],
            'rejected':next(iter(banks.values()))['invalid']})
        files.extend(output/name for name in ('common_scales.pt','questions.json','manifest.json'))
        stage.commit(files,questions=len(questions),banks=2)
        progress.finished_call(len(questions));progress.mark_committed(len(questions))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    parser.add_argument('--reader',choices=['student','teacher'],default='student')
    args=parser.parse_args();build(read_json(args.config),args.reader)
