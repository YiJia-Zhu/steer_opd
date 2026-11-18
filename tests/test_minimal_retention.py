"""Retention boundaries: immutable SV evidence, failed batches and durable saves."""
from pathlib import Path

import pytest
from steer_opd.artifact_reuse import StageArtifacts
from steer_opd.resumable_core import ROOT,atomic_json,read_json
from steer_opd.retention import MINIMAL,minimal
from steer_opd.stage_io import configure_layout,component_target,project
from steer_opd.sv_retention import RECEIPT,retire_pool
from steer_opd.training.retention import (batch_root,reset_batch_workspace,commit_checkpoint,
    validate_checkpoint,prune_checkpoints,recover_checkpoint)
from test_stage_reuse import synthetic_campaign,codecs  # CPU-only artifact fixtures


@pytest.fixture
def plan(tmp_path):
    return {'binding':'test_binding','output':str(tmp_path),'retention_policy':MINIMAL,
            'settings':{'gpu_pool':[0,1]}}


def test_policy_does_not_adopt_historical_runs():
    assert not minimal({}) and minimal({'retention_policy':MINIMAL})
    with pytest.raises(ValueError,match='retention'):minimal({'retention_policy':'typo'})


def test_new_plan_keeps_save_cadence_and_scalar_monitoring_but_disables_archives():
    from steer_opd.training.verl_plan import VerlSettings,build_verl_plan,verl_overrides
    cfg=read_json(ROOT/'configs/ablation.json')
    plan=build_verl_plan(cfg,VerlSettings(monitor_evaluation=True,monitor_overlap=True))
    result=verl_overrides(plan)
    assert plan['retention_policy']==MINIMAL and plan['schema'].endswith('_v7')
    assert result['trainer.save_freq']==20 and result['trainer.test_freq']==16
    assert result['trainer.logger']==['console','tensorboard']
    assert result['trainer.max_actor_ckpt_to_keep'] is None  # project rotates after commit
    assert result['trainer.rollout_data_dir'] is None and result['trainer.validation_data_dir'] is None
    assert result['trainer.log_val_generations']==0
    assert result['actor_rollout_ref.rollout.engine_kwargs.vllm.max_logprobs']==16
    assert 'src/steer_opd/retention.py' in plan['implementation_sha256']


def test_batch_storage_is_bounded_and_keeps_other_outputs(tmp_path,plan):
    folder=tmp_path/'scorer'
    legacy=folder/'requests/old.json';atomic_json(legacy,{'old':'frozen'})
    reset_batch_workspace(folder,plan)
    work=batch_root(folder,plan)
    current=work/'cache/teacher/request/observed.pt';current.parent.mkdir(parents=True)
    current.write_bytes(b'current batch')
    # The unfinished batch stays available until a successful update/restart.
    assert current.exists()
    stats=reset_batch_workspace(folder,plan)
    assert stats['bytes']>=len(b'current batch')
    assert not current.exists() and legacy.exists()
    assert list(work.iterdir())==[work/'owner.json']
    reset_batch_workspace(folder,plan)
    assert legacy.exists()


def test_batch_cleanup_recovers_interrupted_deletion_and_empty_initialization(tmp_path,plan):
    folder=tmp_path/'scorer';reset_batch_workspace(folder,plan)
    work=batch_root(folder,plan);(work/'payload').write_bytes(b'pending')
    work.rename(folder/'.batch_gc_interrupted')
    work.mkdir()  # Crash before the next workspace owner was written.
    reset_batch_workspace(folder,plan)
    assert not list(folder.glob('.batch_gc_*'))
    assert read_json(work/'owner.json')['plan_binding']==plan['binding']


@pytest.mark.parametrize('attack',['owner','symlink'])
def test_batch_cleanup_refuses_foreign_or_linked_data(tmp_path,plan,attack):
    folder=tmp_path/'scorer';reset_batch_workspace(folder,plan)
    work=batch_root(folder,plan);outside=tmp_path/'valuable';outside.write_text('keep')
    if attack=='owner':atomic_json(work/'owner.json',{'plan_binding':'another_run'})
    else:(work/'linked').symlink_to(outside)
    with pytest.raises(ValueError):reset_batch_workspace(folder,plan)
    assert outside.read_text()=='keep' and work.exists()


def test_batch_release_requires_completed_handover_and_pending_update(tmp_path,plan):
    from steer_opd.training.scorer_client import ScorerClient
    client=object.__new__(ScorerClient);client.plan=plan;client.folder=tmp_path/'scorer'
    client.roles=['teacher'];client.states={'teacher':'awake'};client.shared_phase=False
    client.pending_request={'request_id':'batch','payload_sha256':'sha'}
    reset_batch_workspace(client.folder,plan)
    with pytest.raises(RuntimeError):client.release_batch(1)
    client.states['teacher']='asleep';client.shared_phase=True
    with pytest.raises(RuntimeError):client.release_batch(1)
    client.shared_phase=False;client.release_batch(1)
    assert client.pending_request is None


def write_checkpoint(root,step):
    folder=root/f'global_step_{step}';(folder/'actor').mkdir(parents=True)
    (folder/'data.pt').write_bytes(b'dataloader')
    for rank in (0,1):
        for kind in ('model','optim','extra_state'):
            (folder/'actor'/f'{kind}_world_size_2_rank_{rank}.pt').write_bytes(f'{step}-{kind}-{rank}'.encode())
    (root/'latest_checkpointed_iteration.txt').write_text(str(step))
    return folder


def test_checkpoint_rotation_keeps_old_until_all_components_commit(tmp_path,plan):
    root=tmp_path/'checkpoints';old=write_checkpoint(root,20);commit_checkpoint(root,20,plan)
    new=write_checkpoint(root,40);missing=new/'actor/optim_world_size_2_rank_1.pt';missing.unlink()
    with pytest.raises(ValueError,match='Incomplete'):commit_checkpoint(root,40,plan)
    assert old.exists() and recover_checkpoint(root,plan)==old
    assert (root/'latest_checkpointed_iteration.txt').read_text()=='20'
    missing.write_bytes(b'optimizer');(root/'latest_checkpointed_iteration.txt').write_text('40')
    receipt=commit_checkpoint(root,40,plan)
    assert old.exists() and new.exists()
    prune_checkpoints(root,40,plan,committed=receipt)
    assert not old.exists() and recover_checkpoint(root,plan)==new
    (new/'actor/model_world_size_2_rank_0.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='changed'):validate_checkpoint(new,plan)


def test_first_checkpoint_interruption_does_not_become_a_restore_point(tmp_path,plan):
    root=tmp_path/'checkpoints';folder=write_checkpoint(root,20)
    assert recover_checkpoint(root,plan) is None
    assert not (root/'latest_checkpointed_iteration.txt').exists()
    (root/'latest_checkpointed_iteration.txt').write_text('20')
    commit_checkpoint(root,20,plan);(root/'latest_complete.json').unlink()
    assert recover_checkpoint(root,plan)==folder


def test_checkpoint_rotation_does_not_delete_foreign_or_uncommitted_folders(tmp_path,plan):
    root=tmp_path/'checkpoints'
    foreign=write_checkpoint(root,10);commit_checkpoint(root,10,{**plan,'binding':'other'})
    unfinished=write_checkpoint(root,15)
    write_checkpoint(root,20);receipt=commit_checkpoint(root,20,plan)
    prune_checkpoints(root,20,plan,committed=receipt)
    assert foreign.exists() and unfinished.exists()


def minimal_campaign(cfg,codecs,calls):
    campaign=synthetic_campaign(cfg,codecs,calls)
    original=campaign.gpu
    def gpu(name,count,command,*,stage_output,**kwargs):
        original(name,count,command,stage_output=stage_output,**kwargs)
        if name.startswith('activations'):
            path=project(stage_output)/'complete.json'
            atomic_json(path,{**read_json(path),'retention_policy':MINIMAL})
    campaign.gpu=gpu
    return campaign


@pytest.mark.parametrize('paired',[False,True])
def test_retired_pool_resumes_and_reuses_banks_without_gpu(tmp_path,codecs,paired):
    cfg=configure_layout({**read_json(ROOT/'configs/ablation.json'),
        'include_teacher_ablation':paired,'output_root':str(tmp_path/'outputs'),'param_name':'retention_test'})
    calls=[];q=[{'id':'q1','question':'Q','answer':'0'}]
    first=minimal_campaign(cfg,codecs,calls)
    for role in ('teacher','student'):first.generation(role,q)
    first.directions(q);before=list(calls)
    roles=['student','teacher'] if paired else ['student']
    for role in roles:
        folder=project(first.records['capture_'+role]['owner'])
        assert not (folder/'pooled.pt').exists() and (folder/RECEIPT).exists()
        StageArtifacts(cfg).validate(ROOT/first.records['capture_'+role]['descriptor_path'])
    second_cfg=configure_layout({**cfg,'alpha':.3,'param_name':'retention_test_second'})
    second=minimal_campaign(second_cfg,codecs,calls)
    for role in ('teacher','student'):second.generation(role,q)
    second.directions(q)
    assert calls==before and second.cfg['allowed_question_ids']==['q1']
    for role in roles:
        assert first.records['bank_'+role]['artifact_id']==second.records['bank_'+role]['artifact_id']
    # Retiring a pool must not turn missing retained bank data into a cache hit.
    bank_folder=project(first.records['bank_student']['owner']);(bank_folder/'result.json').write_text('corrupt')
    with pytest.raises(ValueError,match='changed'):
        StageArtifacts(cfg).validate(ROOT/first.records['capture_student']['descriptor_path'])


def test_retirement_is_resumable_and_missing_pool_without_receipt_is_rejected(tmp_path,codecs,monkeypatch):
    cfg=configure_layout({**read_json(ROOT/'configs/ablation.json'),
        'output_root':str(tmp_path/'outputs'),'param_name':'retention_interrupt'})
    calls=[];q=[{'id':'q1','question':'Q','answer':'0'}];campaign=minimal_campaign(cfg,codecs,calls)
    original=Path.unlink
    def fail_pool(path,*args,**kwargs):
        if path.name=='pooled.pt':raise OSError('interrupted cleanup')
        return original(path,*args,**kwargs)
    for role in ('teacher','student'):campaign.generation(role,q)
    with monkeypatch.context() as patch:
        patch.setattr(Path,'unlink',fail_pool)
        with pytest.raises(OSError,match='interrupted'):campaign.directions(q)
    capture=campaign.records['capture_student'];folder=project(capture['owner'])
    assert (folder/'pooled.pt').exists() and (folder/RECEIPT).exists()
    campaign.directions(q)
    assert not (folder/'pooled.pt').exists()
    saved=read_json(folder/RECEIPT);saved['usable_ids']=[];atomic_json(folder/RECEIPT,saved)
    with pytest.raises(ValueError,match='retirement identity'):
        StageArtifacts(cfg).validate(ROOT/capture['descriptor_path'])


def test_historical_pool_stays_and_unexplained_missing_file_is_an_error(tmp_path,codecs):
    cfg=configure_layout({**read_json(ROOT/'configs/ablation.json'),
        'output_root':str(tmp_path/'outputs'),'param_name':'historical_pool'})
    q=[{'id':'q1','question':'Q','answer':'0'}]
    campaign=synthetic_campaign(cfg,codecs,[])
    for role in ('teacher','student'):campaign.generation(role,q)
    campaign.directions(q)
    packet=campaign.records['capture_student'];folder=project(packet['owner'])
    assert (folder/'pooled.pt').exists() and not (folder/RECEIPT).exists()
    (folder/'pooled.pt').unlink()
    with pytest.raises(ValueError,match='changed'):
        StageArtifacts(cfg).validate(ROOT/packet['descriptor_path'])
