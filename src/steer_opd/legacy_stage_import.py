"""CPU registration of committed overlap artifacts, preserving their original bytes."""
from pathlib import Path

import torch

from steer_opd import stage_contracts as keys
from steer_opd.artifact_reuse import verify_commit
from steer_opd.compass_campaign import generation_job,make_base_job
from steer_opd.compass_job import prepare_plans,native_score_binding,cached_source_metrics,safe_key
from steer_opd.compass_scoring import AlignedMetricAdapter
from steer_opd.compass_banks import make_common_banks
from steer_opd.direction_bank import file_sha256
from steer_opd.extraction import encode_answer
from steer_opd.math_data import prompt_text
from steer_opd.resumable_core import ROOT,digest,read_json
from steer_opd.rollout_topk import load_rollout_topk
from steer_opd.stage_campaign import Campaign,prepare_data
from steer_opd.stage_io import component_target,project

# Reviewed student-space extension: prompt key generalized and reader role added;
# the student engine, pooling positions, means and norms are identical.
CAPTURE_COMPAT={
 'src/steer_opd/extraction.py':('1ccd82434336bbfc4601faf6d4f191347aeb330dd7f14b99b04d404c20c66aa6',
                              '412980226ef234bbfd6dca99b9ca2d1905134b8ca058fb8f6ba9c11017a2a0ee'),
 'src/steer_opd/easysteer_extraction.py':('659d0bf21a208b5f090e41812e69d32d96293558cce4246d046b5b3253ae6de0',
                                      'f385bc45b58975fe7e1479ce218232856332325aa2ed6be533d89df6fa6046fb')}


def assert_equal(a,b):
    if isinstance(a,torch.Tensor):
        if not isinstance(b,torch.Tensor) or not torch.equal(a,b):raise ValueError('Imported bank tensor changed')
    elif isinstance(a,dict):
        if set(a)!=set(b):raise ValueError('Imported structure changed')
        for k in a:assert_equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        if len(a)!=len(b):raise ValueError('Imported sequence changed')
        for x,y in zip(a,b):assert_equal(x,y)
    elif a!=b:raise ValueError('Imported identity or scalar changed')


def runtime_proof(folder,runtime,code,*,student_capture=False):
    record=read_json(folder/'reproducibility.json');bundle=project(record['source_bundle'])
    snapshot=read_json(bundle)
    if digest(snapshot['sha256'])!=record['source_revision']:raise ValueError('Original code revision changed')
    for name in code:
        original=snapshot['sha256'][name];current=file_sha256(ROOT/name)
        if digest_source(snapshot['source'][name])!=original:raise ValueError('Original source bundle changed')
        if original!=current and not (student_capture and CAPTURE_COMPAT.get(name)==(original,current)):
            raise ValueError(f'Unreviewed implementation change prevents legacy reuse: {name}')
    if any(record['versions'].get(k)!=v for k,v in runtime.items()):
        raise ValueError('Legacy engine packages differ from the current environment')
    return [folder/'reproducibility.json',bundle]


def digest_source(text):
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()


def register(cfg,legacy_run,progress):
    legacy=project(legacy_run)
    old=read_json(legacy/'config.json')
    if old.get('metric_version')!=cfg['metric_version'] or old.get('direction_version')!=cfg['direction_version']:
        raise ValueError('Legacy protocol differs; do not migrate metrics or directions by renaming')
    if cfg.get('include_teacher_ablation'):raise ValueError('Register the original student-only parameter set first')
    for key in ('question_count','teacher','student','seed','enable_thinking','layers','sv_max_context_len'):
        if old[key]!=cfg[key]:raise ValueError(f'Legacy import must describe its original {key}')
    campaign=Campaign(cfg,progress,None,None)
    original_index=legacy/'artifact_index.json';index=read_json(original_index)['files']
    campaign.remember('data',prepare_data(cfg,campaign.codecs))
    questions=read_json(project(campaign.cfg['data_dir'])/'questions.json')
    if file_sha256(project(old['data_dir'])/'questions.json')!=file_sha256(project(campaign.cfg['data_dir'])/'questions.json'):
        raise ValueError('Original questions or native prompts changed')

    def publish(name,science,owner,files,proofs):
        target=component_target(cfg,name)
        found=campaign.store.find(name,science,target)
        if found:result=campaign.store.bind(target,found,science)
        else:
            result=campaign.store.publish_external(name,science,owner,files,target,proofs=proofs,
                provenance={'adapter':'legacy_overlap_top16_v1','original_run':str(legacy.relative_to(ROOT)),
                            'original_code_and_completion_preserved':True,'gpu_executed':False})
        campaign.remember(name,result);progress.event('cache_registered',name=name)

    def committed(owner):
        files=verify_commit(owner)
        return [owner/name for name in files]+[owner/'complete.json',owner/'stage_input.json']

    for role,relative in [('teacher','references/teacher'),('student','source')]:
        owner=legacy/relative
        proofs=runtime_proof(owner,campaign.runtime,keys.GEN_CODE)
        packet=read_json(owner/'job.json');actual=packet['job']
        expected=generation_job(campaign.cfg,campaign.codecs[role],role,questions,source=role=='student')
        ignored={'output','execution_hold','execution_blockers','gpu_acceptance_receipt','gpu_acceptance_binding',
                 'model_content_id','backend_version','metric_version','matching_rule_version',
                 'reuse_committed_texts','reuse_committed_rollout'}
        if packet['hash']!=digest(actual) or actual.get('model_content_id')!=campaign.models[role]:
            raise ValueError('Original generation model/job identity changed')
        if {k:v for k,v in actual.items() if k not in ignored}!={k:v for k,v in expected.items() if k not in ignored}:
            raise ValueError('Original generation requests or parameters differ')
        if read_json(owner/'complete.json')['binding']!=digest({'job':actual,'runtime':read_json(owner/'reproducibility.json')['binding']}):
            raise ValueError('Original generation runtime binding changed')
        for request in expected['requests']:
            row=read_json(owner/(request['key']+'.json'))
            if any(row.get(k)!=v for k,v in request.items()) or row.get('model_content_id')!=campaign.models[role] or row.get('steering'):
                raise ValueError('Original source row identity changed')
            if role=='student':load_rollout_topk(owner,row,16,expected['support_size'])
        science=keys.generation(campaign.cfg,role,questions,campaign.codecs[role],campaign.models[role],campaign.runtime)
        publish(role,science,owner,committed(owner),proofs)

    owner=legacy/'activations';parents={r:campaign.pid(r) for r in ('teacher','student')}
    proofs=runtime_proof(owner,campaign.runtime,keys.CAP_CODE,student_capture=True)
    value=torch.load(owner/'pooled.pt',weights_only=True,map_location='cpu');meta=value['identity']
    tokenizer=campaign.codecs['student'].tokenizer
    required={'checkpoint':cfg['student'],'checkpoint_content_id':campaign.models['student'],
        'source_checkpoint_contents':campaign.models,'layer':cfg['layers'][0],
        'tokenizer_sha256':digest(tokenizer.get_vocab()),'template_sha256':digest(tokenizer.chat_template),
        'enable_thinking':cfg['enable_thinking'],'pooling':'sv_training_all_retained_answer_tokens',
        'sv_max_context_len':cfg['sv_max_context_len'],'direction_version':cfg['direction_version']}
    if any(meta.get(k)!=v for k,v in required.items()) or set(value['pooled'])!={q['id'] for q in questions}:
        raise ValueError('Original pooled checkpoint, protocol, questions or context changed')
    text_identity=[]
    for q in questions:
        pair=[read_json(project(campaign.records[r]['owner'])/(q['id']+('_0' if r=='student' else '')+'.json'))
              for r in ('teacher','student')]
        prompt=prompt_text(tokenizer,q['question'],cfg['enable_thinking'])
        valid=all(encode_answer(tokenizer,prompt,row.get('text'),cfg['sv_max_context_len'])['response_ids'] for row in pair)
        record=value['pooled'][q['id']]
        if bool(record['usable'])!=bool(valid):raise ValueError('Pooled usable set changed')
        if valid:
            if record['identity']['text_record_hashes']!=[digest(row) for row in pair]:raise ValueError('Pooled texts changed')
            for layer in record['layers'].values():
                if len(layer['pools'])!=2 or any(not torch.isfinite(t).all() for t in layer['pools']+layer['norms']):
                    raise ValueError('Invalid pooled means/norms')
        text_identity.append((q['id'],q['question'],pair,bool(valid)))
    if meta['texts_hash']!=digest(text_identity) or read_json(owner/'complete.json')['binding']!=digest(meta):
        raise ValueError('Pooled text/runtime commit changed')
    publish('capture_student',keys.capture(cfg,'student',campaign.models['student'],campaign.codecs['student'],parents,campaign.runtime),
            owner,committed(owner),proofs)

    owner=legacy/'banks';manifest=read_json(owner/'manifest.json')
    expected,_=make_common_banks(value['pooled'],cfg['layers'],manifest['identity'],alpha=manifest['alpha'])
    for name,bank in expected.items():assert_equal(bank,torch.load(owner/(name+'.pt'),weights_only=True,map_location='cpu'))
    common=[q['id'] for q in questions if value['pooled'][q['id']]['usable']]
    publish('bank_student',keys.bank(cfg,'student',campaign.pid('capture_student'),common),owner,committed(owner),
            [owner/'manifest.json',legacy/'config.json'])

    rows=read_json(legacy/'source_rows.json')
    base=make_base_job(campaign.cfg,campaign.codecs,str((legacy/'source_rows.json').relative_to(ROOT)))
    base.update(alignment_dir=str((legacy/'alignment').relative_to(ROOT)),alignment_source_dir=str((legacy/'alignment').relative_to(ROOT)))
    aligned,plans,role_rows,_=prepare_plans(base,campaign.codecs)
    if rows!=[read_json(project(campaign.records['student']['owner'])/(q['id']+'_0.json')) for q in questions]:
        raise ValueError('Original alignment rows differ from the source rollout')
    role_hash=digest(role_rows)
    if role_hash!=read_json(legacy/'alignment/manifest.json')['role_rows_hash']:raise ValueError('Native alignment rows changed')
    alignment_files=[legacy/'alignment/manifest.json',legacy/'alignment/teacher_rows.json',
                     *(legacy/'alignment'/f'{key}.json' for key in plans)]
    for path in alignment_files:
        if file_sha256(path)!=index[str(path.relative_to(legacy))]['sha256']:raise ValueError('Original alignment checksum changed')
    proofs=runtime_proof(legacy/'native_scores/teacher',campaign.runtime,keys.ALIGN_CODE)
    publish('alignment',keys.alignment(cfg,campaign.pid('student'),campaign.codecs),legacy/'alignment',alignment_files,
            [original_index,*proofs])
    owner=legacy/'native_scores/teacher'
    proofs=runtime_proof(owner,campaign.runtime,keys.NATIVE_CODE)
    for row in role_rows['teacher']:
        record=torch.load(owner/(row['key']+'.pt'),weights_only=True,map_location='cpu')
        if record['binding']!=native_score_binding(base,'teacher',row,campaign.codecs['teacher']):
            raise ValueError('Native teacher prefix/checkpoint identity changed')
    science=keys.native(cfg,campaign.models['teacher'],campaign.codecs['teacher'],
        {'student':campaign.pid('student'),'alignment':campaign.pid('alignment')},campaign.runtime)
    publish('native_teacher',science,owner,committed(owner),proofs)

    base.update(source_metrics_dir=str((legacy/'alignment').relative_to(ROOT)),teacher_scores=str(owner.relative_to(ROOT)))
    adapter=AlignedMetricAdapter(plans,campaign.codecs['student'],campaign.codecs['teacher'],require_dual=True)
    metric_files=[]
    for row in aligned:
        key=safe_key(row);folder=legacy/'alignment';marker=folder/(key+'.metrics.json')
        if not marker.exists():raise ValueError('Legacy metrics are incomplete; registration never computes missing results')
        source=project(campaign.records['student']['owner']);teacher=owner/(key+'.pt')
        cached_source_metrics(base,row,load_rollout_topk(source,row,16,base['support_size']),
            torch.load(teacher,weights_only=True,map_location='cpu'),adapter,
            file_sha256(source/row['rollout_topk']),file_sha256(teacher))
        metric_files.extend([marker,folder/(key+'.metrics.pt')])
    publish('source_metrics',keys.metrics({k:campaign.pid(k) for k in ('student','native_teacher','alignment')}),
            legacy/'alignment',metric_files,[original_index])
    register_historical_results(cfg,legacy)
    return {name:{'owner':p['owner'],'artifact_id':p['artifact_id']} for name,p in campaign.records.items()}


def register_historical_results(cfg,legacy):
    legacy=project(legacy)
    historical={split:{'summaries':str((legacy/split/'summaries.json').relative_to(ROOT)),
        'sha256':file_sha256(legacy/split/'summaries.json'),
        'complete_sha256':file_sha256(legacy/split/'complete.json')} for split in ('select','confirm')}
    file=project(cfg['step_dirs']['6'])/'metadata/historical_results.json'
    from steer_opd.compass_campaign import frozen_json
    frozen_json(file,{'scope':'historical_original_code_results_not_current_screen_cache','sources':historical})
