"""Resolve recorded historical artifact paths after an audited physical move."""
from functools import lru_cache
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
LEDGER=ROOT/'audit/stage_output_migration.json'


@lru_cache(maxsize=2)
def _redirects(path,mtime,size):
    value=json.loads(Path(path).read_text())
    if value.get('status')!='complete':return {},[]
    exact={item['source']:item['target'] for item in value['moves']}
    prefixes=sorted(value['directory_redirects'].items(),key=lambda item:len(item[0]),reverse=True)
    return exact,prefixes


def relocate_path(value):
    """Change a path's location only; never rewrite its frozen file contents."""
    p=Path(value);p=p if p.is_absolute() else ROOT/p
    if not LEDGER.exists():return p
    try:relative=str(p.resolve().relative_to(ROOT.resolve()))
    except ValueError:return p
    stat=LEDGER.stat();exact,prefixes=_redirects(str(LEDGER),stat.st_mtime_ns,stat.st_size)
    target=exact.get(relative)
    if target is None:
        for old,new in prefixes:
            if relative==old or relative.startswith(old+'/'):
                target=new+relative[len(old):];break
    if target is None:return p
    result=ROOT/target
    if not result.resolve().is_relative_to(ROOT.resolve()):raise ValueError('Relocated artifact escapes project')
    return result
