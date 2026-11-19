"""Build the training runtime for new launches alongside previously frozen versions."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'src/steer_opd'
EVAL_PATCH = ROOT / 'runtime/training/evaluation_text.patch'
BASELINE_PATCH = ROOT / 'runtime/baselines/native_training.patch'


def contents(directory):
    return {
        str(path.relative_to(directory)): (
            str(path.readlink()) if path.is_symlink() else hashlib.sha256(path.read_bytes()).hexdigest()
        )
        for path in directory.rglob('*') if path.suffix in {'.py', '.yaml'}
    }


def prepare(*, baseline=False):
    cache = ROOT / '.runtime' / ('baselines' if baseline else 'training')
    cache.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=cache) as temporary:
        build = Path(temporary) / 'build'
        package = build / 'src/steer_opd'
        package.mkdir(parents=True)
        shutil.copytree(PACKAGE / 'training', package / 'training', ignore=shutil.ignore_patterns('__pycache__'))
        private_modules = ('experiment_runtime.py', 'evaluation_text.py')
        for name in private_modules:
            shutil.copyfile(PACKAGE / name, package / name)
        # Apply only to private regular files, before linking shared project modules.
        for patch in ((EVAL_PATCH, BASELINE_PATCH) if baseline else (EVAL_PATCH,)):
            subprocess.run(['patch', '--batch', '--forward', '--fuzz=0', '--no-backup-if-mismatch',
                            '-s', '-p1', '-i', str(patch)], cwd=build, check=True)
        for source in PACKAGE.glob('*.py'):
            if source.name not in private_modules:
                (package / source.name).symlink_to(source)

        files = contents(build)
        identity = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
        target = cache / identity
        with (cache / 'publish.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if target.exists():
                if contents(target) != files:
                    raise ValueError(f'Frozen training runtime source changed at {target}')
            else:
                build.rename(target)
    return target / 'src'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', action='store_true')
    print(prepare(baseline=parser.parse_args().baseline))
