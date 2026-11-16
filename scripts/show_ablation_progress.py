#!/usr/bin/env python3
"""Read-only progress/ETA display; no torch, GPU access or experiment startup."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def duration(seconds):
    if seconds is None:return '?'
    seconds=max(0,int(seconds));return f'{seconds//3600:02d}:{seconds//60%60:02d}:{seconds%60:02d}'


def process_alive(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
    except OSError:return False


def progress_roots(cfg):
    roots={ROOT/cfg['run_dir']}
    if cfg.get('layout_version')==1:
        roots.update(ROOT/value for key,value in cfg['step_dirs'].items() if key not in ('4','5'))
        meta=ROOT/cfg['run_dir']/'metadata/upstreams'
        for ref in meta.glob('*.json'):
            try:
                owner=(ROOT/json.loads(ref.read_text())['owner']).resolve()
                if owner.is_relative_to(ROOT.resolve()):roots.add(owner)
            except (OSError,KeyError,ValueError):continue
    return roots


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/ablation.json');p.add_argument('--watch',action='store_true')
    args=p.parse_args();cfg=json.loads((ROOT/args.config).read_text());run=ROOT/cfg['run_dir']
    while True:
        if args.watch and sys.stdout.isatty():print('\033[2J\033[H',end='')
        print(f"Run: {cfg['id']}  experiment={cfg.get('experiment')} phase={cfg.get('phase')} "
              f"hold={cfg.get('execution_hold')}  {datetime.now().astimezone().isoformat()}")
        states=[]
        if (run/'metadata/historical_run.json').exists():
            print('已完成的历史实验；参数更新为0。')
        elif cfg.get('layout_version')==1 and not (run/'logs/progress.json').exists():
            print('本参数尚未执行；下方包含已注册上游的历史进度。')
        seen=set()
        for root in progress_roots(cfg):
            for f in root.rglob('progress.json'):
                if 'attempts' in f.relative_to(root).parts or f.resolve() in seen:continue
                seen.add(f.resolve())
                try:states.append(json.loads(f.read_text()))
                except (OSError,ValueError):continue
        for s in sorted(states,key=lambda x:x['updated_at'],reverse=True):
            status=s['status']
            if status=='running' and not process_alive(s['pid']):status='interrupted (last heartbeat)'
            print(f"{s['stage']}: {status} | {s['completed']}/{s['total']} {s['unit']} | "
                  f"elapsed {duration(s['elapsed_seconds'])} | ETA {duration(s['eta_seconds'])} | phase={s['phase']}\n"
                  f"  started {s['started_at']} | committed {s.get('committed', '?')} | resumed {s['resumed']} | active total {duration(s['active_elapsed_seconds'])}")
        if not states:print('No execution progress yet; experiment has not started.')
        sys.stdout.flush()
        if not args.watch:break
        time.sleep(5)


if __name__=='__main__':main()
