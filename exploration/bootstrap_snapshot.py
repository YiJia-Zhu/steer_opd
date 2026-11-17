from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[1]
identity=json.loads((ROOT/'audit/expanded_search_launch_20261117.json').read_text())['source_revision']
bundle=json.loads((ROOT/'.runtime/source_revisions'/identity/'source_bundle.json').read_text())
for name,body in bundle['source'].items():
 if name.startswith('src/steer_opd/'):
  p=ROOT/'exploration/snapshot'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(body)
p=ROOT/'exploration/snapshot/src/steer_opd/resumable_core.py';p.write_text(p.read_text().replace('ROOT = Path(__file__).resolve().parents[2]',f'ROOT = Path({str(ROOT)!r})'))
p=ROOT/'exploration/snapshot/src/steer_opd/easysteer_static.py';p.write_text(p.read_text().replace('Path(__file__).resolve().parents[2] /',f'Path({str(ROOT)!r}) /'))
p=ROOT/'exploration/run_on_idle_gpus.py';p.write_text((ROOT/'scripts/run_on_idle_gpus.py').read_text().replace('ROOT = Path(__file__).resolve().parents[1]',f'ROOT = Path({str(ROOT)!r})'))
manifest={'frozen_param_source_revision':identity,'copied_from':str((ROOT/'.runtime/source_revisions'/identity/'source_bundle.json').relative_to(ROOT)),
 'isolation':'Only exploration/ files added; shared src/, scripts/, configs/ unchanged',
 'snapshot_files':{str(p.relative_to(ROOT/'exploration')):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'exploration/snapshot').rglob('*.py'))}}
scheduler=ROOT/'exploration/run_on_idle_gpus.py'
s=scheduler.read_text().replace('!= ROOT:', "!= ROOT/'exploration':").replace('ROOT / \".runtime/gpu_leases.json\"','ROOT / \"exploration/.runtime/gpu_leases.json\"').replace('ROOT / \".runtime/gpu_admission.lock\"','ROOT / \"exploration/.runtime/gpu_admission.lock\"').replace('subprocess.Popen(command, cwd=ROOT,',"subprocess.Popen(command, cwd=ROOT/'exploration',")
scheduler.write_text(s)
(ROOT/'exploration/snapshot_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('snapshot',identity,'files',len(manifest['snapshot_files']))
