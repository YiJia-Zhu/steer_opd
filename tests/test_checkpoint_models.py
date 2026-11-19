"""Model preservation and crash-safe retirement of older training resume state."""
from pathlib import Path

import pytest

from steer_opd.retention import MINIMAL,ALL_MODELS_LATEST_STATE
from steer_opd.resumable_core import atomic_json,read_json
from steer_opd.training.retention import (MODEL_RECEIPT,STATE_RETIREMENT,commit_checkpoint,
    validate_checkpoint,validate_model_checkpoint,prune_checkpoints,recover_checkpoint)


@pytest.fixture
def plan(tmp_path):
    return {'binding':'model_history_test','output':str(tmp_path),'retention_policy':MINIMAL,
        'checkpoint_retention':ALL_MODELS_LATEST_STATE,'settings':{'gpu_pool':[0,1]}}


def write_checkpoint(root,step):
    folder=root/f'global_step_{step}'
    (folder/'actor/huggingface').mkdir(parents=True)
    atomic_json(folder/'actor/huggingface/config.json',{'model_type':'synthetic'})
    (folder/'data.pt').write_bytes(b'dataloader')
    for rank in (0,1):
        for kind in ('model','optim','extra_state'):
            (folder/'actor'/f'{kind}_world_size_2_rank_{rank}.pt').write_bytes(f'{step}-{kind}-{rank}'.encode())
    (root/'latest_checkpointed_iteration.txt').write_text(str(step))
    return folder


def test_all_saved_models_survive_and_only_latest_state_is_resumable(tmp_path,plan):
    root=tmp_path/'checkpoints';models={}
    for step in (16,32,48,64):
        folder=write_checkpoint(root,step)
        model=folder/'actor/model_world_size_2_rank_0.pt'
        models[step]=(model.stat().st_ino,model.read_bytes())
        receipt=commit_checkpoint(root,step,plan)
        prune_checkpoints(root,step,plan,committed=receipt)
    assert len(list(root.glob('global_step_*')))==4
    for step,(inode,content) in models.items():
        folder=root/f'global_step_{step}'
        model=folder/'actor/model_world_size_2_rank_0.pt'
        assert model.stat().st_ino==inode and model.read_bytes()==content
        assert (folder/'actor/huggingface/config.json').is_file()
        saved=validate_model_checkpoint(folder,plan)
        assert 'data.pt' not in saved['files']
        assert (folder/'data.pt').exists()==(step==64)
        assert len(list((folder/'actor').glob('optim_*.pt')))==(2 if step==64 else 0)
        assert len(list((folder/'actor').glob('extra_state_*.pt')))==(2 if step==64 else 0)
        if step!=64:
            assert (folder/STATE_RETIREMENT).is_file()
            with pytest.raises(ValueError,match='changed'):validate_checkpoint(folder,plan)
    assert recover_checkpoint(root,plan)==root/'global_step_64'


def test_failed_model_receipt_cannot_retire_previous_complete_save(tmp_path,plan,monkeypatch):
    import steer_opd.training.retention as retention
    root=tmp_path/'checkpoints';old=write_checkpoint(root,16);commit_checkpoint(root,16,plan)
    newer=write_checkpoint(root,32)
    original=retention.atomic_json
    def fail(path,value):
        if path==newer/MODEL_RECEIPT:raise OSError('interrupted model receipt')
        return original(path,value)
    with monkeypatch.context() as patch:
        patch.setattr(retention,'atomic_json',fail)
        with pytest.raises(OSError,match='interrupted'):commit_checkpoint(root,32,plan)
    assert recover_checkpoint(root,plan)==old
    assert (old/'data.pt').is_file() and not (old/STATE_RETIREMENT).exists()
    with pytest.raises(ValueError,match='restore point'):prune_checkpoints(root,32,plan)
    (root/'latest_checkpointed_iteration.txt').write_text('32')
    receipt=commit_checkpoint(root,32,plan);prune_checkpoints(root,32,plan,committed=receipt)
    assert validate_model_checkpoint(old,plan) and recover_checkpoint(root,plan)==newer


def test_state_retirement_recovers_after_partial_unlink(tmp_path,plan,monkeypatch):
    root=tmp_path/'checkpoints';old=write_checkpoint(root,16);commit_checkpoint(root,16,plan)
    newer=write_checkpoint(root,32);receipt=commit_checkpoint(root,32,plan)
    original=Path.unlink
    def fail(path,*args,**kwargs):
        if path==old/'actor/optim_world_size_2_rank_0.pt':raise OSError('interrupted state retirement')
        return original(path,*args,**kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(Path,'unlink',fail)
        with pytest.raises(OSError,match='interrupted'):prune_checkpoints(root,32,plan,committed=receipt)
    assert (old/STATE_RETIREMENT).exists()
    assert validate_model_checkpoint(old,plan)
    assert recover_checkpoint(root,plan)==newer
    prune_checkpoints(root,32,plan,committed=receipt)
    assert not (old/'data.pt').exists() and not list((old/'actor').glob('optim_*.pt'))
    assert validate_model_checkpoint(old,plan)


@pytest.mark.parametrize('damage',['model','optimizer','retirement'])
def test_corruption_blocks_deletion_of_old_recovery_state(tmp_path,plan,damage):
    root=tmp_path/'checkpoints';old=write_checkpoint(root,16);commit_checkpoint(root,16,plan)
    write_checkpoint(root,32);receipt=commit_checkpoint(root,32,plan)
    if damage=='retirement':atomic_json(old/STATE_RETIREMENT,{'binding':'damaged'})
    else:
        kind='model' if damage=='model' else 'optim'
        (old/'actor'/f'{kind}_world_size_2_rank_0.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='changed'):prune_checkpoints(root,32,plan,committed=receipt)
    assert (old/'data.pt').exists()
    assert len(list((old/'actor').glob('extra_state_*.pt')))==2
    assert len(list((old/'actor').glob('optim_*.pt')))==2


def test_rotation_preserves_foreign_legacy_and_uncommitted_checkpoints(tmp_path,plan):
    root=tmp_path/'checkpoints'
    legacy=write_checkpoint(root,4);commit_checkpoint(root,4,{k:v for k,v in plan.items() if k!='checkpoint_retention'})
    foreign=write_checkpoint(root,8);commit_checkpoint(root,8,{**plan,'binding':'other-plan'})
    incomplete=write_checkpoint(root,12)
    write_checkpoint(root,16);receipt=commit_checkpoint(root,16,plan)
    prune_checkpoints(root,16,plan,committed=receipt)
    assert all((folder/'data.pt').is_file() for folder in (legacy,foreign,incomplete))


@pytest.mark.parametrize('field',['plan_binding','receipt_sha256'])
def test_bad_latest_pointer_never_authorizes_state_retirement(tmp_path,plan,field):
    root=tmp_path/'checkpoints';old=write_checkpoint(root,16);commit_checkpoint(root,16,plan)
    write_checkpoint(root,32);receipt=commit_checkpoint(root,32,plan)
    pointer=read_json(root/'latest_complete.json');pointer[field]='changed'
    atomic_json(root/'latest_complete.json',pointer)
    with pytest.raises(ValueError,match='restore point'):prune_checkpoints(root,32,plan,committed=receipt)
    assert (old/'data.pt').is_file() and not (old/STATE_RETIREMENT).exists()


def test_symlinked_checkpoint_state_is_not_cleaned(tmp_path,plan):
    root=tmp_path/'checkpoints';old=write_checkpoint(root,16);commit_checkpoint(root,16,plan)
    write_checkpoint(root,32);receipt=commit_checkpoint(root,32,plan)
    external=tmp_path/'external.pt';external.write_bytes(b'keep external bytes')
    path=old/'actor/optim_world_size_2_rank_0.pt';path.unlink();path.symlink_to(external)
    with pytest.raises(ValueError,match='Symlink'):prune_checkpoints(root,32,plan,committed=receipt)
    assert external.read_bytes()==b'keep external bytes' and (old/'data.pt').is_file()


def test_handover_acks_are_consumed_without_a_history(tmp_path,plan):
    from steer_opd.training.scorer_client import ScorerClient
    client=object.__new__(ScorerClient);client.plan=plan;client.folder=tmp_path/'scorer'
    client.states={'teacher':'asleep'};client.shared_phase=True
    def reply(predicate):
        command=read_json(client.folder/'control/teacher.json')
        atomic_json(client.folder/'acks'/f"{command['command_id']}.json",{
            'command_id':command['command_id'],'role':'teacher','plan_binding':plan['binding'],
            'state':'awake' if command['action']=='wake' else 'asleep'})
        assert predicate()
    client._wait=reply
    client.control('teacher','wake');client.control('teacher','sleep')
    assert client.states=={'teacher':'asleep'} and not list((client.folder/'acks').iterdir())
    assert len(list((client.folder/'control').iterdir()))==1
