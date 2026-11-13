#!/usr/bin/env python3
"""Start a project command after two consecutive idle-GPU observations.

This is admission checking, not a machine-wide resource reservation. Never
terminates any process. GPU UUIDs, command, child PID and exit code are recorded.
"""

import argparse
import csv
import datetime as dt
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]


def process_ticks(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z" or Path(f"/proc/{pid}/cwd").resolve(strict=True) != ROOT:
            return None
        return int(fields[19])
    except (OSError, ValueError, IndexError):
        return None


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f".tmp.{os.getpid()}")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def source_snapshot():
    files = sorted([*(ROOT / "src/steer_opd").glob("*.py"), *(ROOT / "scripts").glob("*.py")])
    bodies = {str(path.relative_to(ROOT)): path.read_text() for path in files}
    hashes = {path: hashlib.sha256(body.encode()).hexdigest() for path, body in bodies.items()}
    identity = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    target = ROOT / ".runtime/source_revisions" / identity / "source_bundle.json"
    if not target.exists():
        write_json(target, {"sha256": hashes, "source": bodies})
    return {"id": identity, "bundle": str(target.relative_to(ROOT))}


def snapshot():
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,name,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True)
    apps = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits",
    ], text=True)
    processes = [(row[0].strip(), int(row[1])) for row in csv.reader(io.StringIO(apps)) if len(row) >= 2]
    occupied = {uuid for uuid, pid in processes}
    project_owned = {uuid for uuid, pid in processes if process_ticks(pid) is not None}
    rows = []
    for row in csv.reader(io.StringIO(output)):
        index, uuid, name, memory, utilization = [value.strip() for value in row]
        rows.append({"index": int(index), "uuid": uuid, "name": name,
                     "memory_mib": int(memory), "utilization": int(utilization),
                     "has_compute_process": uuid in occupied, "project_owned": uuid in project_owned})
    return rows


def idle(row):
    return not row["has_compute_process"] and row["memory_mib"] <= 128 and row["utilization"] <= 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--wait-seconds", type=int, default=0)
    parser.add_argument("--max-project-gpus", type=int, default=2)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or not 1 <= args.count <= args.max_project_gpus <= 2:
        parser.error("A command is required; project GPU limit is1 or2")
    run_dir = args.run_dir.resolve()
    if not run_dir.is_relative_to(ROOT):
        raise ValueError("Run directory must be in the project")
    run_dir.mkdir(parents=True, exist_ok=True)
    record_path = run_dir / "process.json"
    if record_path.exists():
        raise FileExistsError("Use a new run directory; never overwrite an existing process record")
    queue = {"pid": os.getpid(), "start_ticks": process_ticks(os.getpid()), "command": command,
             "created_at": dt.datetime.now(dt.timezone.utc).isoformat(), "status": "waiting_idle_gpu"}
    write_json(run_dir / "queue.json", queue)
    lease_path = ROOT / ".runtime/gpu_leases.json"
    lock_path = ROOT / ".runtime/gpu_admission.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.wait_seconds
    while True:
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            leases = json.loads(lease_path.read_text()) if lease_path.exists() else []
            leases = [lease for lease in leases if process_ticks(lease["pid"]) == lease["start_ticks"] and lease["start_ticks"] is not None]
            first = snapshot()
            time.sleep(2)
            second = snapshot()
            leased = {uuid for lease in leases for uuid in lease["uuids"]}
            own = leased | {row["uuid"] for row in second if row.get("project_owned")}
            earlier = {r["uuid"] for r in first if idle(r)}
            selected = [r for r in second if idle(r) and r["uuid"] in earlier and r["uuid"] not in leased][:args.count]
            if len(own)+args.count > args.max_project_gpus:
                selected = []
            if len(selected) == args.count:
                env = dict(os.environ)
                env["CUDA_VISIBLE_DEVICES"] = ",".join(r["uuid"] for r in selected)
                env["STEER_OPD_GPU_ADMISSION"] = str(record_path)
                record = {"started_at": dt.datetime.now(dt.timezone.utc).isoformat(), "cwd": str(ROOT),
                    "command": command, "gpus": selected, "before": [first, second], "supervisor_pid": os.getpid()}
                record["source_revision"] = source_snapshot()
                log = (run_dir / "stdout.log").open("x")
                try:
                    child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                             stderr=subprocess.STDOUT, start_new_session=True)
                except BaseException:
                    log.close()
                    raise
                record.update(pid=child.pid, child_start_ticks=process_ticks(child.pid))
                write_json(record_path, record)
                leases.append({"pid": child.pid, "start_ticks": record["child_start_ticks"],
                    "uuids": [row["uuid"] for row in selected], "process_record": str(record_path)})
                write_json(lease_path, leases)
                queue.update(status="launched", child_pid=child.pid)
                write_json(run_dir / "queue.json", queue)
                break
            write_json(lease_path, leases)
        if time.monotonic() >= deadline:
            (run_dir / "no_idle_gpu.json").write_text(json.dumps(second, indent=2) + "\n")
            raise SystemExit("Not enough idle GPUs; no process was started or terminated")
        print("Waiting for idle GPUs; no external processes are changed", flush=True)
        time.sleep(10)
    try:
        print(json.dumps({"pid": child.pid, "gpus": selected, "log": str(run_dir / "stdout.log")}), flush=True)
        record["exit_code"] = child.wait()
    finally:
        log.close()
    record["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json(record_path, record)
    raise SystemExit(record["exit_code"])


if __name__ == "__main__":
    main()
