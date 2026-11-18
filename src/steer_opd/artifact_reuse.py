"""Immutable cross-parameter stage references, matched by scientific dependencies."""
from pathlib import Path

from steer_opd.direction_bank import file_sha256
from steer_opd.resumable_core import ROOT, atomic_json, digest, read_json
from steer_opd.stage_io import project

SCHEMA='scientific_stage_artifact_v1'


def code_identity(names):
    return {name:file_sha256(ROOT/name) for name in names}


def verify_files(folder,files):
    for name,item in files.items():
        path=folder/name
        if not path.resolve().is_relative_to(folder.resolve()):raise ValueError('Committed file escapes its owner')
        if not path.is_file() or path.stat().st_size!=item['bytes'] or file_sha256(path)!=item['sha256']:
            raise ValueError(f'Committed file changed: {path}')
    return {name:{'sha256':v['sha256'],'bytes':v['bytes']} for name,v in files.items()}


def verify_commit(folder):
    """Validate an existing whole-engine-call commit without rebinding its identity."""
    folder=project(folder);marker=folder/'complete.json'
    packet=read_json(marker)
    if packet.get('schema')!='engine_call_v1':
        if packet.get('status')!='awaiting_user_review' or not (folder/'summaries.json').exists():
            raise ValueError(f'Not a committed engine call or complete screen: {folder}')
        summaries=read_json(folder/'summaries.json')
        if len(summaries)!=packet.get('candidates'):raise ValueError('Screen candidate count changed')
        names=['baseline',*summaries]
        if packet.get('teacher_drift_recorded'):names.append('teacher_unsteered_replay')
        files={}
        for name in names:
            for file,item in verify_commit(folder/name).items():files[f'{name}/{file}']=item
            p=folder/name/'complete.json'
            files[f'{name}/complete.json']={'sha256':file_sha256(p),'bytes':p.stat().st_size}
        for p in folder.rglob('*'):
            if p.is_file() and p.suffix in ('.png','.svg','.csv','.pdf'):
                files[str(p.relative_to(folder))]={'sha256':file_sha256(p),'bytes':p.stat().st_size}
        for name in ('summaries.json','teacher_repeat_drift.json','manifest.json'):
            p=folder/name
            if p.exists():files[name]={'sha256':file_sha256(p),'bytes':p.stat().st_size}
        return files
    return verify_files(folder,packet['files'])


class StageArtifacts:
    def __init__(self,cfg):
        self.cfg=cfg
        self.root=project(cfg.get('output_root','outputs'))

    def descriptor(self,folder):return project(folder)/'metadata/artifact.json'

    def validate(self,path,contract=None):
        path=project(path);packet=read_json(path)
        if packet.get('descriptor_path')!=str(path.relative_to(ROOT)):
            raise ValueError('Artifact descriptor location changed')
        if packet.get('schema')!=SCHEMA or packet['contract_key']!=digest(packet['contract']):
            raise ValueError('Artifact contract or schema changed')
        if contract is not None and packet['contract_key']!=digest(contract):
            raise ValueError('Artifact scientific dependencies do not match')
        folder=project(packet['owner'])
        receipt=project(packet.get('completion_receipt',str(folder/'complete.json')))
        if file_sha256(receipt)!=packet['complete_sha256']:
            raise ValueError('Artifact completion receipt changed')
        if packet.get('external_snapshot'):
            proof=read_json(receipt)
            if proof!={'schema':'external_stage_snapshot_v1','owner':packet['owner'],'files':packet['files'],
                        'proofs':packet['proofs']}:raise ValueError('Imported stage receipt changed')
            for name,sha in packet['proofs'].items():
                if file_sha256(project(name))!=sha:raise ValueError('Original import provenance changed')
            files=verify_files(folder,packet['files'])
        elif packet['component'].startswith('capture_') and (folder/'metadata/pool_retirement.json').exists():
            from steer_opd.sv_retention import verify_retired_capture
            files=verify_retired_capture(self,path,packet)
        else:files=verify_commit(folder)
        if files!=packet['files']:raise ValueError('Artifact contents differ from the frozen descriptor')
        expected=digest({'contract_key':packet['contract_key'],'files':files})
        if expected!=packet['artifact_id']:raise ValueError('Artifact content identity changed')
        return folder,packet

    def find(self,component,contract,target,*,materialized=False):
        target=project(target);ref=target/'metadata/reuse.json'
        if ref.exists():
            saved=read_json(ref)
            if saved['contract_key']!=digest(contract):raise ValueError('Frozen upstream reference parameters changed')
            if file_sha256(project(saved['descriptor']))!=saved['descriptor_sha256']:
                raise ValueError('Frozen descriptor changed')
            folder,packet=self.validate(saved['descriptor'],contract)
            if project(saved['owner'])!=folder:raise ValueError('Frozen upstream owner changed')
            if packet['artifact_id']!=saved['artifact_id']:raise ValueError('Frozen upstream artifact changed')
            if materialized and (folder/'metadata/pool_retirement.json').exists():
                raise ValueError('This frozen capture was retired; use a new PARAM_NAME to recapture')
            return folder,packet
        own=self.descriptor(target)
        if own.exists():
            found=self.validate(own,contract)
            if materialized and (found[0]/'metadata/pool_retirement.json').exists():
                raise ValueError('This frozen capture was retired; use a new PARAM_NAME to recapture')
            return found
        # Only inspect small descriptor files at the documented parameter depth.
        candidates=sorted(self.root.glob('step_*/*/*/*/*/*/**/metadata/artifact.json'))
        for path in candidates:
            packet=read_json(path)
            if packet.get('component')==component and packet.get('contract_key')==digest(contract):
                if materialized and (project(packet['owner'])/'metadata/pool_retirement.json').exists():continue
                if (materialized and component.startswith('capture_')
                        and read_json(project(packet['owner'])/'complete.json').get('retention_policy')=='minimal_v1'):
                    continue  # Another run's temporary pool may be retired at any time.
                return self.validate(path,contract)
        return None

    def bind(self,target,found,contract):
        folder,packet=found;target=project(target)
        if folder!=target:
            receipt={'schema':'stage_reference_v1','contract_key':digest(contract),
                'descriptor':packet['descriptor_path'],'descriptor_sha256':file_sha256(project(packet['descriptor_path'])),
                'artifact_id':packet['artifact_id'],'owner':str(folder.relative_to(ROOT))}
            path=target/'metadata/reuse.json'
            if path.exists() and read_json(path)!=receipt:raise ValueError('Refuse to change an upstream reference')
            if not path.exists():atomic_json(path,receipt)
        return folder,packet

    def publish(self,component,contract,folder,*,files=None,provenance=None,descriptor_folder=None):
        folder=project(folder)
        if files is not None and not (folder/'complete.json').exists():
            from steer_opd.experiment_runtime import StageCheckpoint
            StageCheckpoint(folder,digest(contract)).commit([project(p) for p in files],component=component)
        committed=verify_commit(folder)
        path=self.descriptor(descriptor_folder or folder)
        packet={'schema':SCHEMA,'component':component,'contract':contract,'contract_key':digest(contract),
            'descriptor_path':str(path.relative_to(ROOT)),
            'owner':str(folder.relative_to(ROOT)),'complete_sha256':file_sha256(folder/'complete.json'),
            'files':committed,'provenance':provenance or {}}
        packet['artifact_id']=digest({'contract_key':packet['contract_key'],'files':committed})
        if path.exists() and read_json(path)!=packet:raise ValueError('Refuse to change a published artifact')
        if not path.exists():atomic_json(path,packet)
        return folder,packet

    def publish_external(self,component,science,owner,files,target,*,proofs,provenance):
        """Register original bytes with a new receipt; never write in the old owner."""
        owner=project(owner);target=project(target)
        entries={str(project(p).relative_to(owner)):{'sha256':file_sha256(project(p)),
            'bytes':project(p).stat().st_size} for p in files}
        verify_files(owner,entries)
        original={str(project(p).relative_to(ROOT)):file_sha256(project(p)) for p in proofs}
        receipt=target/'metadata/import_complete.json'
        value={'schema':'external_stage_snapshot_v1','owner':str(owner.relative_to(ROOT)),
            'files':entries,'proofs':original}
        if receipt.exists() and read_json(receipt)!=value:raise ValueError('Refuse to change imported evidence')
        if not receipt.exists():atomic_json(receipt,value)
        path=self.descriptor(target)
        packet={'schema':SCHEMA,'component':component,'contract':science,'contract_key':digest(science),
            'descriptor_path':str(path.relative_to(ROOT)),'owner':str(owner.relative_to(ROOT)),
            'completion_receipt':str(receipt.relative_to(ROOT)),'complete_sha256':file_sha256(receipt),
            'files':entries,'proofs':original,'external_snapshot':True,'provenance':provenance}
        packet['artifact_id']=digest({'contract_key':packet['contract_key'],'files':entries})
        if path.exists() and read_json(path)!=packet:raise ValueError('Refuse to change an imported artifact')
        if not path.exists():atomic_json(path,packet)
        return self.validate(path,science)


def contract(component,science,*,parents=None,code=()):
    """Names, paths, logging and injection settings are excluded unless used here."""
    return {'version':1,'component':component,'science':science,'parents':parents or {},
            'implementation':code_identity(code)}
