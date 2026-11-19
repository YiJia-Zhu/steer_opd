"""Default project training paths, explicit external paths and quota checks."""
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from steer_opd.resumable_core import ROOT,atomic_json,read_json,digest
from steer_opd.stage_io import layout_paths
from steer_opd.training.paths import training_path,stored_path
from steer_opd.training.storage import quota_remaining,require_training_space
from steer_opd.training.verl_plan import build_verl_plan,VerlSettings,verl_overrides


@pytest.fixture
def experiment(tmp_path):
    cfg=read_json(ROOT/'configs/ablation.json')
    data=tmp_path/'synthetic_data'
    questions=[{'id':f'q{i}','question':f'Synthetic question {i}','answer':'0'} for i in range(1024)]
    atomic_json(data/'questions_source.json',questions)
    return {**cfg,'input_data_dir':str(data),'training_input_data_dir':str(data)}


def test_default_training_uses_project_output_root(experiment):
    assert training_path('~/steer_opd_outputs')==(Path.home()/'steer_opd_outputs').resolve()
    before=layout_paths(experiment)
    plan=build_verl_plan(experiment)
    expected=(ROOT/'outputs/step_4_train/ablation').resolve()
    assert Path(plan['output']).is_relative_to(expected)
    assert plan['settings']['sv_bank'].startswith('outputs/step_2_sv/')
    assert layout_paths(experiment)==before
    overrides=verl_overrides(plan)
    for key in ('trainer.default_local_dir','data.train_files','data.val_files'):
        values=overrides[key] if isinstance(overrides[key],list) else [overrides[key]]
        assert all(Path(v).is_relative_to(Path(plan['output'])) for v in values)


def test_cpu_prepare_and_hydra_support_external_training_directory(experiment,tmp_path,monkeypatch):
    from steer_opd.training.verl_entry import freeze_prepare
    home=tmp_path/'isolated_home';monkeypatch.setattr(Path,'home',lambda:home)
    plan=build_verl_plan({**experiment,'train_output_root':str(home/'steer_opd_outputs')})
    saved=freeze_prepare(plan)
    assert saved==home/'steer_opd_outputs'/Path(plan['output']).relative_to(home/'steer_opd_outputs')/'metadata/plan.json'
    assert read_json(saved)==plan
    config=read_json(saved.parent/'verl_config.json')
    assert Path(config['trainer']['default_local_dir']).is_relative_to(home/'steer_opd_outputs')
    assert (saved.parent.parent/'data/train.parquet').is_file()


def test_path_guard_preserves_project_paths_and_rejects_unapproved_external_paths(tmp_path,monkeypatch):
    import steer_opd.training.paths as paths
    project=tmp_path/'project';home=tmp_path/'home'
    monkeypatch.setattr(paths,'ROOT',project);monkeypatch.setattr(Path,'home',lambda:home)
    assert training_path('outputs/a')==project/'outputs/a'
    assert stored_path(project/'outputs/a')=='outputs/a'
    assert training_path(home/'steer_opd_outputs/run/scorer')==home/'steer_opd_outputs/run/scorer'
    assert Path(stored_path(home/'steer_opd_outputs/run')).is_absolute()
    with pytest.raises(ValueError,match='Training files'):training_path(tmp_path/'unapproved')
    (home/'steer_opd_outputs').mkdir(parents=True)
    (home/'steer_opd_outputs/escape').symlink_to(tmp_path/'unapproved',target_is_directory=True)
    with pytest.raises(ValueError,match='Training files'):training_path(home/'steer_opd_outputs/escape/file')


def test_resume_handover_can_bind_validate_and_retire_home_checkpoint(experiment,tmp_path,monkeypatch):
    import steer_opd.training.paths as paths
    import steer_opd.training.resume_handover as resume
    from steer_opd.training.retention import commit_checkpoint,validate_model_checkpoint
    home=tmp_path/'isolated_home';project=tmp_path/'isolated_project'
    monkeypatch.setattr(Path,'home',lambda:home)
    monkeypatch.setattr(paths,'ROOT',project);monkeypatch.setattr(resume,'ROOT',project)
    experiment={**experiment,'train_output_root':str(home/'steer_opd_outputs')}
    old=build_verl_plan(experiment,VerlSettings(run_name='old'))
    old['assets']={'fixture':'same'};old['binding']=digest({k:v for k,v in old.items() if k!='binding'})
    source=Path(old['output']);checkpoint=source/'checkpoints/global_step_20'
    (checkpoint/'actor').mkdir(parents=True)
    (checkpoint/'data.pt').write_bytes(b'fixture')
    for kind in ('model','optim','extra_state'):
        for rank in (0,1):(checkpoint/'actor'/f'{kind}_world_size_2_rank_{rank}.pt').write_bytes(b'fixture')
    (checkpoint.parent/'latest_checkpointed_iteration.txt').write_text('20')
    commit_checkpoint(checkpoint.parent,20,old)
    atomic_json(source/'metadata/plan.json',old)
    new=build_verl_plan(experiment,VerlSettings(run_name='new'));new['assets']=old['assets']
    resume.bind_resume_source(new,checkpoint)
    assert new['resume_source']['checkpoint']==str(checkpoint)
    assert all(Path(name).is_absolute() for name in new['resume_source']['files'])
    resume.validate_resume_source(new)
    new_root=Path(new['output']);(new_root/'checkpoints').mkdir(parents=True)
    (new_root/'checkpoints/latest_checkpointed_iteration.txt').write_text('40')
    current=new_root/'checkpoints/global_step_40';(current/'actor').mkdir(parents=True)
    (current/'data.pt').write_bytes(b'new dataloader')
    for kind in ('model','optim','extra_state'):
        for rank in (0,1):(current/'actor'/f'{kind}_world_size_2_rank_{rank}.pt').write_bytes(b'new fixture')
    commit_checkpoint(new_root/'checkpoints',40,new)
    resume.retire_resume_source(new,40)
    assert validate_model_checkpoint(checkpoint,old) and not (checkpoint/'data.pt').exists()
    assert not list((checkpoint/'actor').glob('optim_*.pt'))
    assert (new_root/'metadata/resume_source_retired.json').is_file()


@pytest.mark.parametrize('line,expected',[
    (' /dev/vda3 100 0 200 1 0 0',100*1024),
    (' /dev/vda3 100* 150 200 1 0 0',50*1024),
    (' /dev/vda3 250 0 200 1 0 0',0),
    (' /dev/vda3 100 0 0 1 0 0',None)])
def test_quota_parser(line,expected):
    assert quota_remaining('Filesystem blocks quota limit files quota limit\n'+line)==expected


def test_training_preflight_uses_user_quota_not_mount_free_space(tmp_path,monkeypatch):
    import steer_opd.training.storage as storage
    monkeypatch.setattr(storage.shutil,'disk_usage',lambda _:NS(free=800*1024**3))
    monkeypatch.setattr(storage.subprocess,'check_output',lambda *a,**k:json.dumps({
        'filesystems':[{'target':'/','options':'rw,relatime,usrquota'}]}))
    monkeypatch.setattr(storage.subprocess,'run',lambda *a,**k:NS(returncode=0,
        stdout=f'/dev/vda3 {48*1024**2} 0 {137*1024**2} 100 0 0',stderr=''))
    info=require_training_space(tmp_path,48*1024**3)
    assert info['available_bytes']==89*1024**3
    with pytest.raises(OSError,match='capacity/quota'):require_training_space(tmp_path,96*1024**3)


def test_quota_enabled_filesystem_with_unreadable_quota_is_not_bypassed(tmp_path,monkeypatch):
    import steer_opd.training.storage as storage
    monkeypatch.setattr(storage.shutil,'disk_usage',lambda _:NS(free=800*1024**3))
    monkeypatch.setattr(storage.subprocess,'check_output',lambda *a,**k:json.dumps({
        'filesystems':[{'target':'/','options':'rw,usrquota'}]}))
    monkeypatch.setattr(storage.subprocess,'run',lambda *a,**k:NS(returncode=1,stdout='',stderr='quota unavailable'))
    with pytest.raises(RuntimeError,match='quota record'):require_training_space(tmp_path,1024)
