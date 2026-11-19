"""Remove completed scratch data after verified output relocation."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / '.runtime'
AUDIT = ROOT / 'audit/matrix16_20261119/relocation.json'
PRESERVE = {'tmp', 'ray', 'easysteer-plugin', 'easysteer-build', 'verl_training',
            'checkpoint_fingerprints', 'source_revisions', 'remote_storage',
            'gpu_admission.lock', 'gpu_leases.json'}


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def referenced(names):
    matches = {name: set() for name in names}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            if p.stat().st_uid != os.getuid() or p.name == str(os.getpid()):
                continue
            if (p / 'stat').read_text().rsplit(')', 1)[1].split()[0] == 'Z':
                continue
            values = [os.readlink(p / 'cwd'), (p / 'cmdline').read_bytes().decode(errors='replace'),
                      (p / 'environ').read_bytes().decode(errors='replace')]
            for fd in (p / 'fd').iterdir():
                try:
                    values.append(os.readlink(fd))
                except OSError:
                    pass
            values.append((p / 'maps').read_text(errors='replace'))
            for name in names:
                token = 'steer_opd/.runtime/' + name
                if any(token in value for value in values):
                    matches[name].add(int(p.name))
        except OSError:
            continue
    return {name: sorted(pids) for name, pids in matches.items() if pids}


def main():
    audit = json.loads(AUDIT.read_text())
    assert audit['status'] == 'transfer_verified' and audit['artifact_descriptors_verified'] == 19
    check = ROOT / audit['artifact_validation']
    assert sha(check) == audit['artifact_validation_sha256']
    snapshot = ROOT / audit['source_snapshot']
    assert sha(snapshot) == audit['source_snapshot_sha256']
    protected = audit['live_training_files_unchanged']
    assert all(sha(ROOT / path) == value for path, value in protected.items())
    candidates = sorted(p.name for p in RUNTIME.iterdir() if p.name not in PRESERVE)
    blocked = referenced(candidates)
    assert not blocked, f'Runtime paths still in use: {blocked}'
    evidence = ROOT / 'audit/runtime_cleanup_20261119'
    evidence.mkdir(parents=True, exist_ok=True)
    archive = evidence / 'retained_source_and_logs.tar.gz'
    archive_dirs = {'diagnostics', 'logging_preview_20261114', 'sv_training_audit_20261117',
                    'overlap_update', 'reservation_admission_check_1794933150574179194'}
    def keep(info):
        return None if '__pycache__' in Path(info.name).parts or info.name.endswith(('.pyc', '.pyo')) else info
    with tarfile.open(archive, 'w:gz', compresslevel=3, dereference=False) as bundle:
        for name in candidates:
            path = RUNTIME / name
            if path.is_symlink():
                continue
            if path.is_file() and path.suffix != '.lock' or name in archive_dirs:
                bundle.add(path, arcname=name, filter=keep)
            if name == 'cpu_parallel_work':
                for child in path.iterdir():
                    if child.name in {'src', 'scripts', 'configs', 'tests', 'original_files'} or child.is_file():
                        bundle.add(child, arcname='cpu_parallel_work/' + child.name, filter=keep)
    worktree = RUNTIME / 'worktrees/per_sample_k4'
    assert not referenced(['worktrees', 'm16'])
    if worktree.exists():
        subprocess.run(['git', 'worktree', 'remove', '--force', str(worktree)], cwd=ROOT, check=True)
    removed = []
    for name in candidates:
        path = RUNTIME / name
        assert path.parent == RUNTIME and name not in PRESERVE
        if name == 'worktrees':
            path.rmdir()  # Do not recurse into any additional worktree.
        elif path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
        removed.append(str(path.relative_to(ROOT)))
    assert all(sha(ROOT / path) == value for path, value in protected.items())
    audit.update(status='relocated_and_cleaned', cleanup_complete=True,
                 cleanup_time=datetime.datetime.now().astimezone().isoformat(),
                 removed_runtime_entries=removed,
                 retained_runtime_entries=sorted(p.name for p in RUNTIME.iterdir()),
                 retained_legacy_evidence=str(archive.relative_to(ROOT)),
                 retained_legacy_evidence_sha256=sha(archive),
                 training_master_alive_after_cleanup=Path('/proc/2698073').exists(),
                 live_training_source_config_and_bank_unchanged_after_cleanup=True)
    text = json.dumps(audit, ensure_ascii=False, indent=2) + '\n'
    AUDIT.write_text(text)
    summary = ROOT / Path(audit['manifest']).parent
    (summary / 'receipt.json').write_text(text)
    print(json.dumps({'removed_entries': len(removed), 'retained_runtime': audit['retained_runtime_entries'],
                      'source_snapshot': audit['source_snapshot'], 'training_master_alive': audit['training_master_alive_after_cleanup'],
                      'live_training_files_unchanged': True, 'remote_mount_touched': False}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
