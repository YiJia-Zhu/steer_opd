#!/usr/bin/env python3
"""Preview pinned experiment assets; download only with --execute.

Only huggingface_hub is needed for execution. This does not install training
dependencies, execute model code, preprocess datasets, or launch training.
"""
import argparse
import json
import shlex
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'experiments/download_manifest.json')
    parser.add_argument('--profile', action='append', help='Repeat to combine profiles; default: pilot-r1')
    parser.add_argument('--root', type=Path, default=ROOT / 'assets')
    parser.add_argument('--kind', choices=['all', 'model', 'dataset'], default='all')
    parser.add_argument('--list', action='store_true', help='List profiles and exit')
    parser.add_argument('--execute', action='store_true', help='Actually download; otherwise preview only')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if args.list:
        for name, profile in manifest['profiles'].items():
            print(f"{name}: {profile['description']}")
        return

    profiles = args.profile or ['pilot-r1']
    unknown = set(profiles) - manifest['profiles'].keys()
    if unknown:
        parser.error(f'Unknown profile(s): {sorted(unknown)}; use --list')
    selected_keys = {key for profile in profiles for key in manifest['profiles'][profile]['assets']}
    selected = [a for a in manifest['assets'] if a['key'] in selected_keys
                and (args.kind == 'all' or a['repo_type'] == args.kind)]
    weight_bytes = sum(a.get('weight_bytes', 0) for a in selected)
    print(f"Mode: {'DOWNLOAD' if args.execute else 'PREVIEW (no files downloaded)'}")
    print(f"Profiles: {', '.join(profiles)}; assets: {len(selected)}")
    print(f"Selected model safetensors: {weight_bytes / 1e9:.2f} GB (excludes dataset/metadata/cache)")
    for asset in selected:
        destination = args.root / ('models' if asset['repo_type'] == 'model' else 'datasets') / asset['repo_id']
        command = ['hf', 'download', asset['repo_id'], '--repo-type', asset['repo_type'],
                   '--revision', asset['revision'], '--local-dir', str(destination),
                   '--include', *asset['allow_patterns']]
        print(f"\n[{asset['key']}] access={asset.get('access', False)}")
        print(shlex.join(command))
        if 'dataset_settings' in asset:
            print('Processing after download: ' + json.dumps(asset['dataset_settings'], ensure_ascii=False))
    if not args.execute:
        print('\nAdd --execute to download. Dataset filtering and splits are separate steps.')
        return

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        parser.error('Execution needs huggingface_hub. Install it in your chosen download environment.')

    for asset in selected:
        destination = args.root / ('models' if asset['repo_type'] == 'model' else 'datasets') / asset['repo_id']
        # Let Hub authentication resolve locally; never print access tokens.
        snapshot_download(repo_id=asset['repo_id'], repo_type=asset['repo_type'],
                          revision=asset['revision'], local_dir=destination,
                          allow_patterns=asset['allow_patterns'])
        receipt = {'completed_at': datetime.now(timezone.utc).isoformat(),
                   'key': asset['key'], 'repo_id': asset['repo_id'],
                   'repo_type': asset['repo_type'], 'revision': asset['revision'],
                   'destination': str(destination.resolve()),
                   'allow_patterns': asset['allow_patterns'],
                   'scope': 'Snapshot files only; no dataset preprocessing or training.'}
        receipt_dir = args.root / 'receipts'
        receipt_dir.mkdir(parents=True, exist_ok=True)
        (receipt_dir / (asset['key'] + '.json')).write_text(json.dumps(receipt, indent=2) + '\n')
    print('\nSelected snapshots downloaded. Apply dataset processing rules before training.')


if __name__ == '__main__':
    main()
