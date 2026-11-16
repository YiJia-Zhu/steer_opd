"""Physical migration protects content, original inodes and provenance paths."""
from pathlib import Path

import pytest

from steer_opd.resumable_core import atomic_json
from steer_opd.direction_bank import file_sha256
from steer_opd import artifact_paths as paths
from steer_opd import output_migration as migration


def test_recorded_paths_redirect_without_rewriting_frozen_json(tmp_path,monkeypatch):
    monkeypatch.setattr(paths,'ROOT',tmp_path);monkeypatch.setattr(paths,'LEDGER',tmp_path/'ledger.json')
    target=tmp_path/'outputs/teacher/job.json';atomic_json(target,{'output':'runs/old/teacher'})
    before=target.read_bytes()
    atomic_json(paths.LEDGER,{'status':'complete','moves':[{'source':'runs/old/teacher/job.json',
        'target':'outputs/teacher/job.json'}],'directory_redirects':{'runs/old/teacher':'outputs/teacher'}})
    assert paths.relocate_path('runs/old/teacher/job.json')==target
    assert paths.relocate_path(tmp_path/'runs/old/teacher')==tmp_path/'outputs/teacher'
    assert target.read_bytes()==before and not (tmp_path/'runs').exists()
    assert paths.relocate_path('assets/model.json')==tmp_path/'assets/model.json'


def test_incomplete_ledger_does_not_activate_path_redirects(tmp_path,monkeypatch):
    monkeypatch.setattr(paths,'ROOT',tmp_path);monkeypatch.setattr(paths,'LEDGER',tmp_path/'ledger.json')
    atomic_json(paths.LEDGER,{'status':'moving','moves':[], 'directory_redirects':{'runs/old':'outputs/new'}})
    assert paths.relocate_path('runs/old')==tmp_path/'runs/old'


def test_path_redirect_cannot_escape_project(tmp_path,monkeypatch):
    monkeypatch.setattr(paths,'ROOT',tmp_path);monkeypatch.setattr(paths,'LEDGER',tmp_path/'ledger.json')
    atomic_json(paths.LEDGER,{'status':'complete','moves':[], 'directory_redirects':{'runs/old':'../escape'}})
    with pytest.raises(ValueError,match='escapes'):paths.relocate_path('runs/old')


def plan():
    return {'moves':[{'source':'old/payload.pt','target':'outputs/payload.pt'}],
            'delete':[],'source_scopes':['old'],'directory_redirects':{'old':'outputs'}}


def test_move_uses_rename_keeps_inode_and_removes_only_identical_duplicate(tmp_path,monkeypatch):
    monkeypatch.setattr(migration,'ROOT',tmp_path)
    source=tmp_path/'old/payload.pt';source.parent.mkdir();source.write_bytes(b'original tensor bytes')
    dest=tmp_path/'outputs/payload.pt';dest.parent.mkdir();dest.write_bytes(source.read_bytes())
    inode=source.stat().st_ino;science_sha=file_sha256(source);record=plan()
    migration.preflight(record);migration.move_files(record)
    assert dest.stat().st_ino==inode and file_sha256(dest)==science_sha
    assert not source.parent.exists() and record['moves'][0]['duplicate_bytes_removed']==len(dest.read_bytes())
    migration.move_files(record) # interrupted-migration resume is idempotent


def test_different_destination_rejected_before_any_mutation(tmp_path,monkeypatch):
    monkeypatch.setattr(migration,'ROOT',tmp_path)
    source=tmp_path/'old/payload.pt';source.parent.mkdir();source.write_bytes(b'keep source')
    dest=tmp_path/'outputs/payload.pt';dest.parent.mkdir();dest.write_bytes(b'keep destination')
    with pytest.raises(ValueError,match='different data'):migration.preflight(plan())
    assert source.read_bytes()==b'keep source' and dest.read_bytes()==b'keep destination'
