#!/usr/bin/env python3
"""Start a project command after two consecutive idle-GPU observations.

This is admission checking, not a machine-wide resource reservation. Never
terminates any process. GPU UUIDs, command, child PID and exit code are recorded.
"""

import argparse
import csv
import datetime as dt
import io
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]


def snapshot():
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,name,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True)
    apps = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits",
    ], text=True)
    occupied = {row[0].strip() for row in csv.reader(io.StringIO(apps)) if row}
    rows = []
    for row in csv.reader(io.StringIO(output)):
        index, uuid, name, memory, utilization = [value.strip() for value in row]
        rows.append({"index": int(index), "uuid": uuid, "name": name,
                     "memory_mib": int(memory), "utilization": int(utilization),
                     "has_compute_process": uuid in occupied})
    return rows


def idle(row):
    return not row["has_compute_process"] and row["memory_mib"] <= 128 and row["utilization"] <= 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--wait-seconds", type=int, default=0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.count < 1:
        parser.error("A command and positive GPU count are required")
    run_dir = args.run_dir.resolve()
    if not run_dir.is_relative_to(ROOT):
        raise ValueError("Run directory must be in the project")
    run_dir.mkdir(parents=True, exist_ok=True)
    record_path = run_dir / "process.json"
    if record_path.exists():
        raise FileExistsError("Use a new run directory; never overwrite an existing process record")
    deadline = time.monotonic() + args.wait_seconds
    while True:
        first = snapshot()
        time.sleep(2)
        second = snapshot()
        earlier = {r["uuid"] for r in first if idle(r)}
        selected = [r for r in second if idle(r) and r["uuid"] in earlier][:args.count]
        if len(selected) == args.count:
            break
        if time.monotonic() >= deadline:
            (run_dir / "no_idle_gpu.json").write_text(json.dumps(second, indent=2) + "\n")
            raise SystemExit("Not enough idle GPUs; no process was started or terminated")
        print("Waiting for idle GPUs; no external processes are changed", flush=True)
        time.sleep(10)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(r["uuid"] for r in selected)
    env["STEER_OPD_GPU_ADMISSION"] = str(record_path)
    record = {"started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
              "cwd": str(ROOT), "command": command, "gpus": selected,
              "before": [first, second], "supervisor_pid": os.getpid()}
    with (run_dir / "stdout.log").open("x") as log:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        record["pid"] = child.pid
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps({"pid": child.pid, "gpus": selected, "log": str(run_dir / "stdout.log")}), flush=True)
        record["exit_code"] = child.wait()
    record["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    raise SystemExit(record["exit_code"])


if __name__ == "__main__":
    main()
