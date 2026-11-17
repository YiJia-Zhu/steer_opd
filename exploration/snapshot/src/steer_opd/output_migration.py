"""Physically relocate the two completed panels, preserving every original byte."""
from pathlib import Path
import os
import shutil

from steer_opd.artifact_reuse import StageArtifacts,contract,verify_commit
from steer_opd.direction_bank import file_sha256
from steer_opd.resumable_core import ROOT,atomic_json,digest,read_json
from steer_opd.stage_io import configure_layout,component_target,project

LEDGER=ROOT/'audit/stage_output_migration.json'
COMPLETED={100:'overlap4_v1',1000:'overlap4_1000_v1'}
OBSOLETE=['runs/static_dual_overlap_20261114','runs/static_panel_v1',
    'runs/ablation/overlap/prepared','runs/ablation/overlap/prepared_anneal',
    'runs/ablation/overlap/source_overlap_v1','logs/source_overlap_v1_launcher.log',
    '.runtime/source_overlap_v1_launcher.json']


def configs():
    base=read_json(ROOT/'configs/ablation.json');result={}
    for count,name in COMPLETED.items():
        result[count]=configure_layout({**base,'question_count':count,'run_name':'ablation','param_name':None})
    return result


def files(folder):
    return sorted(p for p in folder.rglob('*') if p.is_file()) if folder.is_dir() else [folder]


def build_plan():
    settings=configs();moves=[];redirects={};scopes=[]
    def group(source,target):
        source=ROOT/source;target=ROOT/target
        if not source.exists():return
        if source.is_dir():redirects[str(source.relative_to(ROOT))]=str(target.relative_to(ROOT))
        for path in files(source):
            dest=target/path.relative_to(source) if source.is_dir() else target
            moves.append({'source':str(path.relative_to(ROOT)),'target':str(dest.relative_to(ROOT))})
    for count,name in COMPLETED.items():
        cfg=settings[count];run=ROOT/'runs/ablation/overlap'/name
        if read_json(run/'campaign.json')['status']!='awaiting_user_review':raise ValueError('Panel is not complete')
        for split in ('select','confirm'):verify_commit(run/split)
        step={int(k):Path(v) for k,v in cfg['step_dirs'].items()}
        olddata=Path('data/ablation')/('overlap' if count==100 else 'overlap_1000')
        group(olddata,step[0]);scopes.extend([str(run.relative_to(ROOT)),str(olddata)])
        mapping={
            'references/teacher':step[1]/'teacher','source':step[1]/'student',
            'activations':step[2]/'reader_student/activations','banks':step[2]/'reader_student/banks',
            'native_scores':step[3]/'native_scores','select':step[3]/'select','confirm':step[3]/'confirm',
            'select_alignment':step[3]/'alignment/select','confirm_alignment':step[3]/'alignment/confirm',
            'jobs':step[3]/'metadata/legacy/jobs','launches':step[3]/'metadata/legacy/launches',
            'acceptance':step[3]/'metadata/legacy/acceptance','provenance':step[3]/'metadata/provenance',
            'attempts':step[3]/'logs/attempts'}
        for src,dst in mapping.items():group(run.relative_to(ROOT)/src,dst)
        alignment=run/'alignment'
        redirects[str(alignment.relative_to(ROOT))]=str(step[3]/'alignment')
        for path in files(alignment):
            parent='source_metrics' if '.metrics.' in path.name else 'alignment'
            moves.append({'source':str(path.relative_to(ROOT)),
                          'target':str(step[3]/parent/path.relative_to(alignment))})
        handled={Path(name).parts[0] for name in mapping}|{'alignment'}
        for path in run.iterdir():
            if path.name in handled:continue
            if path.is_dir():raise ValueError(f'Unclassified original directory: {path}')
            if path.name in ('thresholds.json','threshold.json'):dst=step[3]/'calibration'/path.name
            elif path.suffix in ('.log','.jsonl') or path.name=='progress.json':dst=step[3]/'logs'/path.name
            elif path.name in ('source_rows.json','select_rows.json','confirm_rows.json','selected_schemes.json','campaign.json'):
                dst=step[3]/path.name
            else:dst=step[3]/'metadata/legacy'/path.name
            group(path.relative_to(ROOT),dst)
        redirects[str(run.relative_to(ROOT))]=str(step[3]/'metadata/legacy')
        group(Path('logs')/(name+'_launcher.log'),step[3]/'logs/launcher.log')
        group(Path('.runtime')/(name+'_launcher.json'),step[3]/'metadata/legacy/launcher_process.json')
    comparison=configure_layout({**settings[1000],'param_name':'n100_vs_n1000_slerp0.1_layer26_seed42_v1'})['step_dirs']['6']
    group('audit/overlap_100_vs_1000_analysis.md',Path(comparison)/'metadata/original_analysis.md')
    if len({m['source'] for m in moves})!=len(moves):raise ValueError('A source file was assigned twice')
    if len({m['target'] for m in moves})!=len(moves):raise ValueError('Destination collision in migration plan')
    return {'schema':'physical_stage_migration_v1','status':'planned','moves':moves,
            'directory_redirects':redirects,'source_scopes':scopes,'delete':OBSOLETE,
            'gpu_executed':False,'configurations':settings}


def mapped(value,plan):
    p=Path(value);p=p if p.is_absolute() else ROOT/p
    try:key=str(p.resolve().relative_to(ROOT.resolve()))
    except ValueError:return value
    exact={m['source']:m['target'] for m in plan['moves']}
    if key in exact:return exact[key]
    for old,new in sorted(plan['directory_redirects'].items(),key=lambda item:len(item[0]),reverse=True):
        if key==old or key.startswith(old+'/'):return new+key[len(old):]
    return key


def preflight(plan):
    """Read/hash first; reject surprises before moving or deleting anything."""
    for i,item in enumerate(plan['moves']):
        src=ROOT/item['source'];dst=ROOT/item['target'];stat=src.stat()
        item.update(sha256=file_sha256(src),bytes=stat.st_size,inode=stat.st_ino,device=stat.st_dev)
        if dst.exists() and file_sha256(dst)!=item['sha256']:
            # Only CPU-registration checkpoint receipts can collide with originals.
            if 'provenance/checkpoints/' not in item['target'] or dst.stat().st_size>1024*1024:
                raise ValueError(f'Destination contains different data: {dst}')
            item['existing_registration_target']=item['target']
        parent=dst.parent
        while not parent.exists():parent=parent.parent
        if parent.stat().st_dev!=stat.st_dev:raise ValueError('Migration must use same-filesystem rename, not copy')
        if i%3000==0:print(f'Preflight: {i}/{len(plan["moves"])} files',flush=True)
    plan['preserved_bytes']=sum(m['bytes'] for m in plan['moves'])
    plan['deletion_inventory']={}
    for name in plan['delete']:
        path=ROOT/name
        if path.exists():plan['deletion_inventory'][name]={'files':len(files(path)),'bytes':sum(p.stat().st_size for p in files(path))}


def move_files(plan):
    for item in plan['moves']:
        src=ROOT/item['source'];dst=ROOT/item['target']
        if not src.exists():
            if dst.exists() and file_sha256(dst)==item['sha256']:continue
            raise ValueError(f'Neither original nor verified destination exists: {src}')
        dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists():
            if file_sha256(dst)==item['sha256']:
                item['duplicate_bytes_removed']=dst.stat().st_size;dst.unlink()
            else:
                backup=dst.parent.parent.parent/'metadata/layout_registration'/dst.parent.name/dst.name
                backup.parent.mkdir(parents=True,exist_ok=True)
                if backup.exists():raise ValueError('Registration receipt backup already exists')
                dst.rename(backup);item['registration_receipt_preserved_at']=str(backup.relative_to(ROOT))
        src.rename(dst)
        if dst.stat().st_ino!=item['inode']:raise ValueError('Move did not preserve original inode')
    for scope in plan['source_scopes']:
        root=ROOT/scope
        for path,dirs,names in os.walk(root,topdown=False):
            folder=Path(path)
            if not any(folder.iterdir()):folder.rmdir()


def rebase_registered_metadata(plan):
    """Only registration wrappers change; original payloads/manifests stay frozen."""
    for path in (ROOT/'outputs').rglob('metadata/artifact.json'):
        packet=read_json(path)
        if packet.get('schema')!='scientific_stage_artifact_v1':continue
        if not packet.get('external_snapshot'):continue
        old_owner=packet['owner'];packet['owner']=mapped(old_owner,plan)
        # source_metrics were split out of the original alignment directory.
        if packet['component']=='source_metrics':packet['owner']=str(path.parent.parent.relative_to(ROOT))
        packet['proofs']={mapped(k,plan):v for k,v in packet['proofs'].items()}
        receipt=ROOT/packet['completion_receipt']
        value={'schema':'external_stage_snapshot_v1','owner':packet['owner'],'files':packet['files'],'proofs':packet['proofs']}
        atomic_json(receipt,value);packet['complete_sha256']=file_sha256(receipt)
        packet['provenance'].setdefault('physically_relocated_from',old_owner)
        atomic_json(path,packet)
    for path in (ROOT/'outputs').rglob('metadata/upstreams/*.json'):
        value=read_json(path);descriptor=ROOT/value['descriptor'];packet=read_json(descriptor)
        value.update(owner=packet['owner'],descriptor_sha256=file_sha256(descriptor))
        atomic_json(path,value)
    for path in (ROOT/'outputs').rglob('metadata/reuse.json'):
        value=read_json(path);packet=read_json(ROOT/value['descriptor'])
        value.update(owner=packet['owner'],descriptor_sha256=file_sha256(ROOT/value['descriptor']))
        atomic_json(path,value)
    for path in (ROOT/'outputs').rglob('metadata/historical_results.json'):
        value=read_json(path)
        for item in value['sources'].values():item['summaries']=mapped(item['summaries'],plan)
        atomic_json(path,value)


def register_historical(cfg,count,plan):
    """Publish immutable historical descriptors without claiming new GPU validation."""
    from steer_opd.stage_campaign import data_contract,write_summary
    from steer_opd.compass_job import load_codecs
    store=StageArtifacts(cfg);step3=ROOT/cfg['step_dirs']['3'];records={}
    step0=ROOT/cfg['step_dirs']['0']
    if not (step0/'sampling_manifest.json').exists():
        atomic_json(step0/'sampling_manifest.json',{'schema':'original_frozen_100_question_pool_v1',
            'questions':count,'preserved_base_questions':100,'added_questions':0,
            'base_source_sha256':file_sha256(step0/'questions_source.json'),
            'question_source_sha256':file_sha256(step0/'questions_source.json'),
            'seed':42,'original_input_bytes_preserved':True})
    data_descriptor=step0/'metadata/artifact.json'
    if not data_descriptor.exists():
        science=data_contract(cfg,load_codecs(cfg))
        records['data']=store.publish('data',science,step0,files=[step0/n for n in
            ('questions_source.json','questions.json','split.json','sampling_manifest.json')])[1]
    else:records['data']=read_json(data_descriptor)
    components=['teacher','student','capture_student','bank_student','alignment','native_teacher','source_metrics']
    for name in components:
        folder=ROOT/component_target(cfg,name);descriptor=folder/'metadata/artifact.json'
        if descriptor.exists():records[name]=store.validate(descriptor)[1];continue
        if (folder/'complete.json').exists():
            own=[folder/n for n in verify_commit(folder)]+[folder/'complete.json',folder/'stage_input.json']
        elif name=='alignment':own=list(folder.glob('*.json'))
        else:own=[*folder.glob('*.metrics.pt'),*folder.glob('*.metrics.json')]
        science=contract(name,{'historical_only':True,'original_run':COMPLETED[count],
            'original_artifact_index_sha256':file_sha256(step3/'metadata/legacy/artifact_index.json')})
        records[name]=store.publish_external(name,science,folder,own,folder,
            proofs=[step3/'metadata/legacy/artifact_index.json'],
            provenance={'historical_only':True,'no_current_code_gpu_claim':True})[1]
    original=read_json(step3/'metadata/legacy/config.json')
    old_cfg={**cfg,'confirm_top_n':original['confirm_top_n']}
    atomic_json(step3/'metadata/historical_run.json',{'status':'complete_historical_panel','samples':count,
        'original_run':COMPLETED[count],'select_schemes':4,'confirm_schemes':2 if count==100 else 4,
        'parameter_updates':0,'current_code_gpu_validated':False,
        'original_config':'metadata/legacy/config.json'})
    for split in ('select','confirm'):
        folder=step3/split;science=contract('historical_screen_'+split,{'original_run':COMPLETED[count],
            'summary_sha256':file_sha256(folder/'summaries.json')})
        records['historical_screen_'+split]=store.publish('historical_screen_'+split,science,folder,
            provenance={'historical_only':True,'parameter_updates':0})[1]
    summaries={s:read_json(step3/s/'summaries.json') for s in ('select','confirm')}
    write_summary(old_cfg,summaries,records)
    old_cfg['id']=f"ablation_overlap_{old_cfg['param_name']}"
    atomic_json(step3/'metadata/layout_config.json',old_cfg)
    for name,packet in records.items():
        atomic_json(step3/'metadata/upstreams'/f'{name}.json',{'artifact_id':packet['artifact_id'],
            'contract_key':packet['contract_key'],'owner':packet['owner'],'descriptor':packet['descriptor_path'],
            'descriptor_sha256':file_sha256(ROOT/packet['descriptor_path'])})
    return records


def execute(plan):
    if LEDGER.exists():
        plan=read_json(LEDGER)
        if plan.get('all_original_file_hashes_verified'):
            print('Migration already complete; original payloads remain frozen.',flush=True);return
    else:
        preflight(plan);plan['status']='moving';atomic_json(LEDGER,plan)
    move_files(plan);rebase_registered_metadata(plan)
    # Activate only after all payload paths and registration wrappers are correct.
    plan['status']='complete';atomic_json(LEDGER,plan)
    settings={int(k):v for k,v in plan['configurations'].items()}
    for cfg in settings.values():
        cfg['input_data_dir']=cfg['step_dirs']['0']
        cfg['base_question_source']=settings[100]['step_dirs']['0']+'/questions_source.json'
    for count,cfg in settings.items():register_historical(cfg,count,plan)
    for i,item in enumerate(plan['moves']):
        dst=ROOT/item['target']
        if dst.stat().st_size!=item['bytes'] or file_sha256(dst)!=item['sha256']:
            raise ValueError(f'Relocated original bytes changed: {dst}')
        if i%4000==0:print(f'Verified moved bytes: {i}/{len(plan["moves"])}',flush=True)
    for path in (ROOT/'outputs').rglob('metadata/artifact.json'):StageArtifacts(settings[1000]).validate(path)
    # Delete only the explicit failed/prepared experiments after preserved results validate.
    for name in plan['delete']:
        path=ROOT/name
        if path.is_dir():shutil.rmtree(path)
        elif path.exists():path.unlink()
    for name in ('runs/ablation/overlap','runs/ablation','runs','data/ablation','data','logs'):
        path=ROOT/name
        if path.exists() and not any(path.iterdir()):path.rmdir()
    plan.update(status='complete',all_original_file_hashes_verified=True,
        deleted_bytes=sum(v['bytes'] for v in plan['deletion_inventory'].values()),
        duplicate_bytes_removed=sum(m.get('duplicate_bytes_removed',0) for m in plan['moves']))
    atomic_json(LEDGER,plan)
    print({'moved_files':len(plan['moves']),'preserved_bytes':plan['preserved_bytes'],
           'deleted_bytes':plan['deleted_bytes'],'gpu_executed':False},flush=True)
