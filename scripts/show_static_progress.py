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


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/static_dual_overlap.json');p.add_argument('--watch',action='store_true')
    args=p.parse_args();cfg=json.loads((ROOT/args.config).read_text());run=ROOT/cfg['run_dir']
    while True:
        if args.watch and sys.stdout.isatty():print('\033[2J\033[H',end='')
        print(f"Run: {cfg['id']}  hold={cfg.get('execution_hold')}  {datetime.now().astimezone().isoformat()}")
        states=[]
        for f in run.rglob('progress.json'):
            if 'attempts' in f.relative_to(run).parts:continue
            try:states.append(json.loads(f.read_text()))
            except (OSError,ValueError):continue
        for s in sorted(states,key=lambda x:x['updated_at'],reverse=True):
            status=s['status']
            if status=='running' and not Path(f"/proc/{s['pid']}").exists():status='interrupted (last heartbeat)'
            print(f"{s['stage']}: {status} | {s['completed']}/{s['total']} {s['unit']} | "
                  f"elapsed {duration(s['elapsed_seconds'])} | ETA {duration(s['eta_seconds'])} | phase={s['phase']}\n"
                  f"  started {s['started_at']} | committed {s.get('committed', '?')} | resumed {s['resumed']} | active total {duration(s['active_elapsed_seconds'])}")
        if not states:print('No execution progress yet; experiment has not started.')
        sys.stdout.flush()
        if not args.watch:break
        time.sleep(5)


if __name__=='__main__':main()
