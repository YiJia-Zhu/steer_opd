"""One-shot verified relocation; never overwrite live-project files."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]
WORKTREE = ROOT / '.runtime/worktrees/per_sample_k4'
NAME = 'matrix16_n1024_ST_L29_a0.1_53a97cdb70_v1'
SUFFIX = Path('ablation/T_Qwen3.5-35B-A3B_S_Qwen3-4B/OpenThoughts3-1.2M/shared')
SUMMARY = ROOT / 'outputs/step_6_summary' / SUFFIX / NAME
AUDIT = ROOT / 'audit/matrix16_20261119/relocation.json'


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def main():
    progress = json.loads((WORKTREE / SUMMARY.relative_to(ROOT) / 'progress.json').read_text())
    assert progress['status'] == 'complete' and not progress['active'] and not progress['failed']
    protected = list((ROOT / 'src').rglob('*.py')) + list((ROOT / 'scripts').glob('*.py'))
    protected += [ROOT / 'outputs/step_4_train' / SUFFIX / 'opd_n1024_project_outputs_v1/metadata/plan.json']
    source_cfg = json.loads((WORKTREE / '.matrix_authorization.json').read_text())['source_config']
    protected += [ROOT / source_cfg]
    cfg = json.loads((ROOT / source_cfg).read_text())
    protected += [ROOT / 'outputs/step_2_sv' / SUFFIX / cfg['param_name'] / 'reader_teacher/banks/S-T_global.pt']
    before = {str(p.relative_to(ROOT)): sha(p) for p in protected}
    files = sorted(p for p in (WORKTREE / 'outputs').rglob('*') if p.is_file())
    assert not any(p.is_symlink() for p in (WORKTREE / 'outputs').rglob('*'))
    assert shutil.disk_usage(ROOT).free > sum(p.stat().st_size for p in files if not (ROOT / p.relative_to(WORKTREE)).exists()) + 100 * 2**30
    print('Comparing all pre-existing output files; none will be overwritten.', flush=True)
    verified = {}
    for p in files:
        q = ROOT / p.relative_to(WORKTREE)
        if not q.exists():
            continue
        assert q.is_file() and p.stat().st_size == q.stat().st_size, str(q)
        value = sha(p)
        assert os.path.samefile(p, q) or value == sha(q), str(q)
        verified[str(p.relative_to(WORKTREE))] = value
    implementation = SUMMARY / 'metadata/implementation'
    implementation.mkdir(parents=True, exist_ok=True)
    diff = subprocess.run(['git', '-C', str(WORKTREE), 'diff', '--binary', 'HEAD'], capture_output=True, check=True).stdout
    (implementation / 'worktree.patch').write_bytes(diff)
    archive = implementation / 'source_snapshot.tar.gz'
    excluded = {'.git', 'outputs', '.runtime', '.cache', '.pytest_cache', '.venv-opd', '.venv-easysteer', 'output_remote'}
    def keep(info):
        parts = Path(info.name).parts
        return None if '__pycache__' in parts or info.name.endswith(('.pyc', '.pyo')) else info
    with tarfile.open(archive.with_suffix('.tmp'), 'w:gz', compresslevel=3, dereference=False) as bundle:
        for item in sorted(WORKTREE.iterdir()):
            if item.name not in excluded:
                bundle.add(item, arcname=item.name, filter=keep)
    archive.with_suffix('.tmp').replace(archive)
    manifest = SUMMARY / 'metadata/relocation/files.jsonl'
    manifest.parent.mkdir(parents=True, exist_ok=True)
    copied = copied_bytes = total_bytes = 0
    last = time.monotonic()
    with manifest.with_suffix('.tmp').open('w') as stream:
        for index, p in enumerate(files, 1):
            relative = str(p.relative_to(WORKTREE))
            q = ROOT / relative
            size = p.stat().st_size
            value = verified.get(relative)
            operation = 'existing_verified'
            if value is None:
                value = sha(p)
                q.parent.mkdir(parents=True, exist_ok=True)
                temporary = q.with_name(q.name + f'.matrix16_copy_{os.getpid()}')
                shutil.copy2(p, temporary)
                assert temporary.stat().st_size == size and sha(temporary) == value, relative
                with temporary.open('rb') as incoming:
                    os.fsync(incoming.fileno())
                os.link(temporary, q)  # O_EXCL-like publication: never replace an existing file.
                temporary.unlink()
                copied += 1
                copied_bytes += size
                operation = 'copied_verified'
            total_bytes += size
            stream.write(json.dumps({'path': relative, 'bytes': size, 'sha256': value, 'operation': operation}, ensure_ascii=False) + '\n')
            if time.monotonic() - last > 15:
                print(f'Verified {index}/{len(files)} files; copied {copied_bytes / 2**30:.2f} GiB.', flush=True)
                last = time.monotonic()
        stream.flush()
        os.fsync(stream.fileno())
    manifest.with_suffix('.tmp').replace(manifest)
    after = {str(p.relative_to(ROOT)): sha(p) for p in protected}
    assert before == after, 'Live training source/config/bank changed during relocation.'
    result = {'status': 'transfer_verified', 'time': datetime.datetime.now().astimezone().isoformat(),
              'old_project_root': str(WORKTREE), 'new_project_root': str(ROOT),
              'matrix': NAME, 'path_policy': 'Identical project-relative outputs paths; frozen files unchanged.',
              'files_verified': len(files), 'bytes_verified': total_bytes,
              'files_copied': copied, 'bytes_copied': copied_bytes,
              'existing_files_preserved': len(verified), 'live_training_files_unchanged': before,
              'manifest': str(manifest.relative_to(ROOT)), 'manifest_sha256': sha(manifest),
              'source_snapshot': str(archive.relative_to(ROOT)), 'source_snapshot_sha256': sha(archive),
              'cleanup_complete': False, 'remote_mount_touched': False}
    save(AUDIT, result)
    save(SUMMARY / 'metadata/relocation/receipt.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'live_training_files_unchanged'}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
