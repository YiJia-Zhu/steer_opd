#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
host=i2s-remote
src="$project_dir/outputs/step_4_train/ablation/T_Qwen3.5-35B-A3B_S_Qwen3-4B/OpenThoughts3-1.2M/shared"
backup_root="data/working_tmp/steer_opd/backups"
resume=false
if [[ $# == 2 && "$1" == --resume && "$2" =~ ^static_sv_[0-9]{8}_[0-9]{6}_[0-9]+$ ]]; then
    resume=true
    name="$2"
elif [[ $# == 0 ]]; then
    name="static_sv_$(date +%Y%m%d_%H%M%S)_$$"
else
    printf '用法 bash scripts/backup_static_sv.sh [--resume static_sv_YYYYMMDD_HHMMSS_PID]\n' >&2
    exit 2
fi
dest="$backup_root/$name"
stage="${dest}.partial"
run16=opd_n1024_project_outputs_utf8_v2
run32=opd_n1024_project_outputs_verbatim_v3
local16="$src/$run16/checkpoints/global_step_16"
local32="$src/$run32/checkpoints/global_step_32"

require_idle_sources() {
    local processes
    processes=$(ps -ww -eo args=)
    if awk -v r16="$run16" -v r32="$run32" '
        /-m steer_opd[.]training[.]verl_entry/ && (index($0,r16) || index($0,r32)) {found=1}
        END {exit !found}
    ' <<< "$processes"; then
        printf '静态 SV 源运行仍有训练进程，停止归档。\n' >&2
        return 1
    fi
}

source_snapshot() {
    python3 - "$local16" "$local32" <<'PY'
import json
from pathlib import Path
import sys

receipts = ("model_checkpoint.complete.json", "checkpoint.complete.json")
snapshot = []
for checkpoint, receipt in zip(map(Path, sys.argv[1:]), receipts, strict=True):
    if checkpoint.resolve() != checkpoint:
        raise SystemExit(f"Unexpected symlink in checkpoint path: {checkpoint}")
    for path in (checkpoint, *sorted(checkpoint.rglob("*"))):
        if path.is_symlink():
            raise SystemExit(f"Unexpected symlink in checkpoint: {path}")
        stat = path.stat()
        snapshot.append((str(path), stat.st_dev, stat.st_ino, stat.st_size,
                         stat.st_mtime_ns, stat.st_ctime_ns))
    if not (checkpoint / receipt).is_file():
        raise SystemExit(f"Missing committed checkpoint receipt: {checkpoint / receipt}")
print(json.dumps(snapshot))
PY
}

run_manifest() {
    python3 - "$1" "$2" "$3" <<'PY'
import hashlib
import json
from pathlib import Path
import re
import sys

run, step, receipt_name = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
checkpoints = run / "checkpoints"
checkpoint = checkpoints / f"global_step_{step}"
receipt = checkpoint / receipt_name
entries = json.loads(receipt.read_text())["files"]
metadata_names = {
    "checkpoint.complete.json", "model_checkpoint.complete.json", "state_retirement.json"
}
actual = {p.relative_to(checkpoint).as_posix() for p in checkpoint.rglob("*") if p.is_file()}
if not entries or actual - metadata_names != entries.keys():
    raise SystemExit(f"Checkpoint files differ from the committed receipt: {checkpoint}")
manifest = []

def add(path, expected_size=None, expected_sha=None):
    name = path.relative_to(run).as_posix()
    if path.resolve() != path or not path.is_file() or not re.fullmatch(r"[a-zA-Z0-9_./-]+", name):
        raise SystemExit(f"Unexpected source path: {path}")
    size = path.stat().st_size
    if expected_size is not None and size != expected_size:
        raise SystemExit(f"Local checkpoint file size changed: {path}")
    sha = expected_sha or hashlib.sha256(path.read_bytes()).hexdigest()
    manifest.append((name, size, sha))

for name, metadata in sorted(entries.items()):
    add(checkpoint / name, metadata["bytes"], metadata["sha256"])
for name in sorted(actual & metadata_names):
    add(checkpoint / name)
for name in ("plan.json", "verl_config.json", "data_manifest.json"):
    add(run / "metadata" / name)
for name, sha in sorted(json.loads((run / "metadata/data_manifest.json").read_text())["files"].items()):
    add(run / "data" / name, expected_sha=sha)
if receipt_name == "checkpoint.complete.json":
    latest = checkpoints / "latest_complete.json"
    tracker = checkpoints / "latest_checkpointed_iteration.txt"
    pointer = json.loads(latest.read_text())
    if (pointer["step"] != step or tracker.read_text().strip() != str(step)
            or pointer["receipt_sha256"] != hashlib.sha256(receipt.read_bytes()).hexdigest()):
        raise SystemExit(f"Expected the local resume pointers to identify step {step}")
    add(latest)
    add(tracker)
print(json.dumps(manifest))
PY
}

remote_missing() {
    local remote_run=$1 manifest=$2 verifier
    verifier=$(cat <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path.cwd()
for name, size, checksum in json.load(sys.stdin):
    path = root / name
    if path.resolve() != path or (path.exists() and not path.is_file()):
        raise SystemExit(f"Unexpected destination path: {path}")
    if not path.is_file() or path.stat().st_size != size:
        print(name)
        continue
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != checksum:
        print(name)
PY
)
    printf '%s\n' "$manifest" |
        ssh "$host" "cd '$remote_run' && python3 -c '$verifier'"
}

backup_run() {
    local run=$1 step=$2 receipt=$3
    local local_run="$src/$run" remote_run="$stage/$run"
    local manifest missing file parent
    local -a missing_files
    manifest=$(run_manifest "$local_run" "$step" "$receipt")
    ssh "$host" "mkdir -p -- '$remote_run'"
    missing=$(remote_missing "$remote_run" "$manifest")
    if [[ -n "$missing" ]]; then
        mapfile -t missing_files <<< "$missing"
        for file in "${missing_files[@]}"; do
            parent="${file%/*}"
            printf '补传 %s/%s\n' "$run" "$file"
            ssh "$host" "mkdir -p -- '$remote_run/$parent'"
            scp -p "$local_run/$file" "$host:$remote_run/$file"
        done
        missing=$(remote_missing "$remote_run" "$manifest")
    fi
    if [[ -n "$missing" ]]; then
        printf '远端校验失败 %s\n%s\n' "$run" "$missing" >&2
        return 1
    fi
    printf '已校验 %s 第 %s 步\n' "$run" "$step"
}

require_idle_sources
before=$(source_snapshot)
if "$resume"; then
    ssh "$host" "test -d '$stage' && test ! -L '$stage' && test ! -e '$dest'"
else
    ssh "$host" "mkdir -p -- '$backup_root' && mkdir -m 700 -- '$stage'"
fi
backup_run "$run16" 16 model_checkpoint.complete.json
backup_run "$run32" 32 checkpoint.complete.json
ssh "$host" "test ! -e '$dest' && mv -T -- '$stage' '$dest'"
printf '远端备份已校验 %s:~/%s\n' "$host" "$dest"

require_idle_sources
after=$(source_snapshot)
if [[ "$before" != "$after" ]]; then
    printf '本地 checkpoint 在备份期间发生变化，保留本地文件。\n' >&2
    exit 1
fi
rm -rf -- "$local16" "$local32"
printf '本地静态 SV 第 16 和 32 步 checkpoint 已删除。\n'
