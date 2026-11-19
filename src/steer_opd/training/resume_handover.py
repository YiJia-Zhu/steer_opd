"""Explicit checkpoint transfer when adding monitoring to an already running plan."""
from pathlib import Path
import shutil

from steer_opd.direction_bank import file_sha256
from steer_opd.resumable_core import ROOT,read_json,atomic_json,digest
from steer_opd.training.paths import training_path,stored_path,is_retirable_training_checkpoint
from steer_opd.retention import keep_model_history


def bind_resume_source(plan,checkpoint):
    path=training_path(checkpoint)
    if not path.name.startswith('global_step_'):
        raise ValueError('Resume checkpoint must be an existing approved training step')
    source_root=path.parent.parent;source=read_json(source_root/'metadata/plan.json')
    if source['binding']!=digest({k:v for k,v in source.items() if k!='binding'}):raise ValueError('Source plan changed')
    if (source_root/'metadata/invalid_sampling.json').exists():raise ValueError('Cannot resume an invalid-sampling experiment')
    for name in ('student','teacher','training_loss','input_sha256','seed','request_seed_policy','direction','granularity','gate','strength_policy','on_policy'):
        if source[name]!=plan[name]:raise ValueError(f'Resume scientific contract differs: {name}')
    if source.get('student_logprob_source')!=plan.get('student_logprob_source'):
        raise ValueError('Resume scientific contract differs: student_logprob_source')
    if source.get('sampling_policy')!=plan.get('sampling_policy'):
        raise ValueError('Resume scientific contract differs: presence penalty sampling policy')
    for name in ('data_policy','train_split','train_questions','training_ids_sha256'):
        if source.get(name)!=plan.get(name):raise ValueError(f'Resume training question pool differs: {name}')
    ignore={'run_name','monitor_overlap'}
    old={k:v for k,v in source['settings'].items() if k not in ignore}
    new={k:v for k,v in plan['settings'].items() if k not in ignore}
    if old!=new or source['assets']!=plan['assets']:raise ValueError('Resume hyperparameters or assets changed')
    for name in ('losses.py','verl_adapter.py','config.py','sleep_engine.py'):
        key='src/steer_opd/training/'+name
        if source['implementation_sha256'][key]!=plan['implementation_sha256'][key]:
            raise ValueError(f'Resume changes training computation: {name}')
    frozen=ROOT/plan['output']/'metadata/plan.json'
    if frozen.exists():
        existing=read_json(frozen).get('resume_source')
        if existing and existing['checkpoint']==stored_path(path):
            plan['resume_source']=existing
            return
    step=int(path.name.removeprefix('global_step_'))
    if not (source_root/'checkpoints/latest_checkpointed_iteration.txt').is_file():raise ValueError('Uncommitted source checkpoint')
    if int((source_root/'checkpoints/latest_checkpointed_iteration.txt').read_text())<step:
        raise ValueError('Source step is not committed')
    required=[path/'data.pt',*(path/'actor'/f'{kind}_world_size_2_rank_{rank}.pt'
        for kind in ('model','optim','extra_state') for rank in (0,1))]
    if any(not p.is_file() for p in required):raise ValueError('Incomplete two-rank resume checkpoint')
    files={stored_path(p):{'sha256':file_sha256(p),'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns}
           for p in required}
    plan['resume_source']={'checkpoint':stored_path(path),'step':step,
        'plan_binding':source['binding'],'source_plan':stored_path(source_root/'metadata/plan.json'),
        'source_plan_sha256':file_sha256(source_root/'metadata/plan.json'),'files':files,
        'change':'Top16 monitoring only; model, optimizer, scheduler, RNG and dataloader restored'}


def validate_resume_source(plan):
    source=plan.get('resume_source')
    if not source:return
    if (ROOT/plan['output']/'checkpoints/latest_checkpointed_iteration.txt').exists():return
    for name,identity in source['files'].items():
        path=ROOT/name
        if not path.is_file() or path.stat().st_size!=identity['size'] or path.stat().st_mtime_ns!=identity['mtime_ns']:
            raise ValueError('Resume checkpoint file changed after its content hash was frozen')
    if file_sha256(ROOT/source['source_plan'])!=source['source_plan_sha256']:raise ValueError('Resume source metadata changed')


def retire_resume_source(plan,step):
    source=plan.get('resume_source')
    if not source or step<=source['step']:return
    root=ROOT/plan['output'];tracker=root/'checkpoints/latest_checkpointed_iteration.txt'
    if not tracker.exists() or int(tracker.read_text())<step:raise ValueError('New checkpoint not committed')
    path=(ROOT/source['checkpoint']).resolve()
    if not is_retirable_training_checkpoint(path) or path.name!=f'global_step_{source["step"]}':
        raise ValueError('Refuse to retire unrelated checkpoint')
    proof=root/'metadata/resume_source_retired.json'
    if proof.exists():return
    if keep_model_history(plan):
        from steer_opd.training.retention import validate_checkpoint,retire_checkpoint_state,MODEL_RECEIPT
        current=root/'checkpoints'/f'global_step_{step}'
        validate_checkpoint(current,plan)
        pointer=read_json(root/'checkpoints/latest_complete.json')
        if (pointer.get('plan_binding')!=plan['binding'] or pointer.get('step')!=step
                or pointer.get('receipt_sha256')!=file_sha256(current/'checkpoint.complete.json')):
            raise ValueError('Replacement checkpoint is not the latest complete save')
        if file_sha256(ROOT/source['source_plan'])!=source['source_plan_sha256']:
            raise ValueError('Resume source metadata changed before state retirement')
        previous=read_json(ROOT/source['source_plan'])
        action='historical_source_preserved'
        if keep_model_history(previous) and (path/MODEL_RECEIPT).exists():
            retire_checkpoint_state(path,previous,replacement={'checkpoint':stored_path(current),'step':step,
                'plan_binding':plan['binding'],'receipt_sha256':file_sha256(current/'checkpoint.complete.json')})
            action='source_state_retired_models_preserved'
        atomic_json(proof,{'source_checkpoint':source['checkpoint'],'retained_model_checkpoint':source['checkpoint'],
            'replaced_by_step':step,'source_plan_binding':source['plan_binding'],
            'current_plan_binding':plan['binding'],'action':action})
        return
    if path.exists():shutil.rmtree(path)
    atomic_json(proof,{'retired_checkpoint':source['checkpoint'],'replaced_by_step':step,
        'source_plan_binding':source['plan_binding'],'current_plan_binding':plan['binding']})
