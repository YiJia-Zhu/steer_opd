#!/usr/bin/env python3
"""Install the pinned EasySteer runtime in a project-only Python3.12 venv.

This never loads models or starts an experiment. Re-running reapplies the fork
Python overlay after pip, as required by EasySteer's official installation.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = 'baecd9698f32aadef68b41536f3db477b513a0d7'
FORK = '7837b07d21c0ddffd043a30001f2c5654290aec7'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install', action='store_true')
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError('Use Python 3.12 to create this isolated wheel environment')
    if not args.install:
        print('Use --install to provision .venv-easysteer; no GPU/model execution is involved')
        return
    work = ROOT / '.runtime/easysteer-build'
    work.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, 'PIP_CACHE_DIR':str(ROOT/'.cache/pip'), 'TMPDIR':str(work),
           'CUDA_VISIBLE_DEVICES':'', 'HF_HUB_OFFLINE':'1',
           'PYTHONPATH':str(ROOT/'src')+os.pathsep+str(ROOT/'.runtime/easysteer-plugin')}
    def run(*command):
        subprocess.run(list(map(str, command)), cwd=ROOT, env=env, check=True)
    target = ROOT / '.venv-easysteer'
    if not target.exists():
        run(sys.executable, '-m', 'venv', target)
    python = target / 'bin/python'
    run(python, '-m', 'pip', 'install', 'vllm==0.29.0', 'transformers==5.16.1', 'matplotlib', 'accelerate')
    receipts = {}
    sources = []
    for repo, revision in [('EasySteer', FRONTEND), ('EasySteer-vllm-v1', FORK)]:
        archive = work / (repo+'-'+revision+'.tar.gz')
        if not archive.exists():
            urllib.request.urlretrieve(f'https://codeload.github.com/ZJU-REAL/{repo}/tar.gz/{revision}', archive)
        receipts[repo] = {'revision':revision,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}
        extracted = work / (repo+'-'+revision)
        if not extracted.exists():
            with tarfile.open(archive) as tar:
                tar.extractall(work, filter='data')
        sources.append(extracted)
    run(python, '-m', 'pip', 'install', sources[0])
    location = subprocess.check_output([str(python), '-c',
        'import importlib.util; print(next(iter(importlib.util.find_spec("vllm").submodule_search_locations)))'], env=env, text=True).strip()
    destination = Path(location)
    if not destination.resolve().is_relative_to(target.resolve()):
        raise RuntimeError('Refusing to overlay a vLLM outside the isolated environment')
    shutil.copytree(sources[1]/'vllm', destination, dirs_exist_ok=True)
    plugin = ROOT / '.runtime/easysteer-plugin/steer_opd_easysteer_plugin-0.1.dist-info'
    plugin.mkdir(parents=True, exist_ok=True)
    (plugin/'METADATA').write_text('Metadata-Version: 2.1\nName: steer-opd-easysteer-plugin\nVersion: 0.1\n')
    (plugin/'entry_points.txt').write_text('[vllm.general_plugins]\nsteer_opd_slerp = steer_opd.easysteer_plugin:register\n')
    env['PYTHONPATH'] = str(ROOT/'src')+os.pathsep+str(plugin.parent)
    env['VLLM_USE_V2_MODEL_RUNNER'] = '1'
    run(python, ROOT/'scripts/check_easysteer_runtime.py')
    receipts.update(vllm_wheel='0.29.0', python=str(python.relative_to(ROOT)), gpu_started=False,
                    frontend=FRONTEND, fork=FORK)
    (ROOT/'ref/easysteer_runtime/installed.json').write_text(json.dumps(receipts,indent=2)+'\n')
    print('Installed isolated EasySteer runtime; GPU parity remains pending')


if __name__ == '__main__':
    main()
