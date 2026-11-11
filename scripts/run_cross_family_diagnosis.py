#!/usr/bin/env python3
"""Run the baseline cross-family diagnosis before steering or OPD experiments."""

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from steer_opd.runtime import verify_runtime


ROOT = Path(__file__).resolve().parents[1]
BEGIN = "<!-- CROSS_FAMILY_RUN_STATUS_BEGIN -->"
END = "<!-- CROSS_FAMILY_RUN_STATUS_END -->"


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def update_agents(status, stage, out):
    path = ROOT / "AGENTS.md"
    original = path.read_text()
    if BEGIN not in original or END not in original:
        return
    message = (f"{BEGIN}\n最近自动更新：{now()}；跨 family 诊断队列状态 `{status}`，"
               f"阶段 `{stage}`。证据：`{out.relative_to(ROOT)}/campaign.json`。\n"
               "队列完成仅表示数值诊断执行完毕，不自动代表 steering 有效或跨 family OPD 已解决。\n"
               f"{END}")
    content = original[:original.index(BEGIN)] + message + original[original.index(END) + len(END):]
    temp = ROOT / ".agents_run_status.tmp"
    temp.write_text(content)
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "experiments/cross_family_overlap_20261110.json")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(ROOT) or (out / "campaign.json").exists():
        raise ValueError("Use a new project-local diagnosis directory")
    runtime = verify_runtime()
    config = json.loads(args.config.read_text())
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    manifest = {"started_at": now(), "status": "waiting_assets", "stage": "assets",
                "runtime": runtime, "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
                "parameter_updates": 0, "stages": []}
    status_path = out / "campaign.json"

    def save():
        status_path.write_text(json.dumps(manifest, indent=2) + "\n")
        update_agents(manifest["status"], manifest["stage"], out)

    save()
    deadline = time.monotonic() + 3600
    while True:
        missing = []
        for spec in config["students"]:
            path = Path(spec["path"])
            path = path if path.is_absolute() else ROOT / path
            index = path / "model.safetensors.index.json"
            tokenizer_present = (path / "tokenizer.json").exists() or (path / "tokenizer.model").exists()
            if not (path / "config.json").exists() or not (path / "tokenizer_config.json").exists() or not tokenizer_present:
                missing.append(spec["key"])
                continue
            files = set(json.loads(index.read_text())["weight_map"].values()) if index.exists() else {"model.safetensors"}
            if any(not (path / f).exists() for f in files):
                missing.append(spec["key"])
        if not missing:
            break
        if time.monotonic() > deadline:
            manifest.update(status="failed", error="Timed out waiting for model assets", missing_models=missing)
            save()
            raise SystemExit(1)
        print(json.dumps({"status": "waiting_assets", "missing_models": missing}), flush=True)
        time.sleep(15)
    stages = [(s["key"], ["--student", s["key"]]) for s in config["students"]]
    stages.append(("teacher_scores", ["--teacher"]))
    for name, phase_args in stages:
        manifest.update(status="running", stage=name)
        record = {"name": name, "started_at": now()}
        manifest["stages"].append(record)
        save()
        command = [sys.executable, "scripts/run_on_idle_gpus.py", "--count", "1",
                   "--run-dir", str(out / "processes" / name), "--wait-seconds", "3600", "--",
                   sys.executable, "scripts/diagnose_cross_family_overlap.py", "--config", str(out / "config.json"),
                   "--output-dir", str(out), *phase_args]
        record["command"] = command
        print(json.dumps({"stage": name, "status": "starting"}), flush=True)
        result = subprocess.run(command, cwd=ROOT)
        record.update(exit_code=result.returncode, finished_at=now())
        if result.returncode:
            manifest["status"] = "failed"
            save()
            raise SystemExit(result.returncode)
        save()
    manifest.update(status="complete", stage="baseline_summary", finished_at=now())
    save()
    print(json.dumps({"status": "complete", "summary": str(out / "summary.json")}), flush=True)


if __name__ == "__main__":
    main()
