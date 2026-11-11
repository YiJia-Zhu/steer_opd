#!/usr/bin/env python3
"""Evaluate the fixed student and teacher concurrently on two admitted GPUs."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "student": "/home/zhenning/data/huggingface_models/DeepSeek-R1-Distill-Qwen-1.5B",
    "teacher": "/home/zhenning/data/huggingface_datasets/JustRL-DeepSeek-1.5B",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=ROOT / "data/math500_pilot_v2/math500.jsonl")
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=31744)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(ROOT):
        raise ValueError("Output must stay in project")
    gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(gpus) != 2 or not os.environ.get("STEER_OPD_GPU_ADMISSION"):
        raise RuntimeError("Two GPUs must be admitted by run_on_idle_gpus.py")
    out.mkdir(parents=True, exist_ok=True)
    processes, logs, records = [], [], {}
    for (role, model), gpu in zip(MODELS.items(), gpus):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["OMP_NUM_THREADS"] = "1"
        command = [sys.executable, "scripts/generate_math.py", "--model", model,
                   "--input", str(args.input),
                   "--output", str(out / f"{role}.jsonl"), "--n", str(args.n),
                   "--max-tokens", str(args.max_tokens)]
        log = (out / f"{role}.log").open("x")
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        processes.append((role, child))
        logs.append(log)
        records[role] = {"pid": child.pid, "gpu_uuid": gpu, "command": command}
    record_path = out / "children.json"
    record_path.write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records, indent=2), flush=True)
    for role, child in processes:
        records[role]["exit_code"] = child.wait()
        record_path.write_text(json.dumps(records, indent=2) + "\n")
    for log in logs:
        log.close()
    raise SystemExit(0 if all(r["exit_code"] == 0 for r in records.values()) else 1)


if __name__ == "__main__":
    main()
