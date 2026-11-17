"""Reuse a committed student baseline when supplementing teacher-only schemes."""
from steer_opd.artifact_reuse import contract,verify_commit
from steer_opd.direction_bank import file_sha256
from steer_opd.resumable_core import ROOT,read_json
from steer_opd.stage_io import metadata_dir,project
from steer_opd.static_masks import validate_static_mask

ADAPTER='historical_direction_metadata_only_v1'


def equivalent_source_mask(old,new,row):
    """Only retired direction metadata may differ; source probabilities/gates may not."""
    validate_static_mask(old,row);validate_static_mask(new,row)
    drop={'direction_version','direction_versions'}
    a={k:v for k,v in old['identity'].items() if k not in drop}
    b={k:v for k,v in new['identity'].items() if k not in drop}
    excluded={'identity','source_binding','mask_id'}
    if a!=b or {k:v for k,v in old.items() if k not in excluded}!={k:v for k,v in new.items() if k not in excluded}:
        raise ValueError('Historical baseline source, metric, threshold or frozen gate changed')


def reference(campaign,split):
    root=project(campaign.cfg['student_baseline_run']);source=root/split
    owner,screen=campaign.store.validate(source/'metadata/artifact.json')
    if owner!=source:raise ValueError('Expected an explicit committed student screen owner')
    for name in ('data','student','alignment','native_teacher','source_metrics'):
        value=read_json(root/'metadata/upstreams'/f'{name}.json')
        if value['artifact_id']!=campaign.pid(name):
            raise ValueError(f'Paired baseline uses a different upstream: {name}')
    baseline=source/'baseline'
    files=[baseline/n for n in verify_commit(baseline)]+[baseline/'complete.json',baseline/'stage_input.json']
    name='paired_student_baseline_'+split
    science=contract(name,{'adapter':ADAPTER,'original_screen':screen['artifact_id'],
        'original_baseline_complete_sha256':file_sha256(baseline/'complete.json')},
        parents={k:campaign.pid(k) for k in ('data','student','alignment','native_teacher','source_metrics')},
        code=('src/steer_opd/paired_baseline.py',))
    target=project(metadata_dir(campaign.cfg))/'paired_baselines'/split
    found=campaign.store.find(name,science,target)
    if found:found=campaign.store.bind(target,found,science)
    else:found=campaign.store.publish_external(name,science,baseline,files,target,
        proofs=[source/'manifest.json',source/'metadata/artifact.json'],
        provenance={'original_student_result_preserved':True,'adapter':ADAPTER})
    campaign.remember(name,found)
    folder,packet=found
    return {'paired_student_baseline':str(folder.relative_to(ROOT)),
            'paired_student_baseline_adapter':ADAPTER,
            'paired_student_baseline_descriptor':packet['descriptor_path'],
            'paired_student_baseline_artifact_id':packet['artifact_id']},packet['artifact_id']
