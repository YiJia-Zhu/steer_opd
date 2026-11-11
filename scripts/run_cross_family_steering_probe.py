#!/usr/bin/env python3
"""Run native calibration and matched-prefix steering diagnosis, without training."""

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from steer_opd.runtime import verify_runtime
from run_cross_family_diagnosis import update_agents


ROOT = Path(__file__).resolve().parents[1]


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(ROOT) or out.exists():
        raise ValueError("Use a new project-local directory")
    if json.loads((args.baseline_dir / "campaign.json").read_text())["status"] != "complete":
        raise ValueError("Finish the unsteered baseline first")
    runtime = verify_runtime()
    config = json.loads(args.config.read_text())
    out.mkdir(parents=True)
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    manifest = {"status": "running", "stage": "teacher_calibration", "started_at": now(),
                "runtime": runtime, "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
                "baseline_dir": str(args.baseline_dir.resolve()), "parameter_updates": 0, "stages": []}

    def save():
        (out / "campaign.json").write_text(json.dumps(manifest, indent=2) + "\n")
        update_agents(manifest["status"], manifest["stage"], out)

    stages = [("teacher_calibration", ["scripts/generate_cross_family_calibration.py",
               "--model-key", "teacher", "--output-dir", str(out / "teacher")])]
    for spec in config["students"]:
        if spec["key"] == "qwen_control":
            continue
        key = spec["key"]
        stages.extend([
            (f"{key}_calibration", ["scripts/generate_cross_family_calibration.py", "--model-key", key,
             "--teacher-answers", str(out / "teacher/calibration.jsonl"), "--output-dir", str(out / key)]),
            (f"{key}_steering", ["scripts/probe_cross_family_steering.py", "--student", key,
             "--baseline-dir", str(args.baseline_dir.resolve()), "--calibration-dir", str(out),
             "--output-dir", str(out / f"{key}_probe")]),
        ])
    save()
    for name, command in stages:
        manifest.update(status="running", stage=name)
        record = {"name": name, "started_at": now()}
        manifest["stages"].append(record)
        save()
        full_command = [sys.executable, "scripts/run_on_idle_gpus.py", "--count", "1", "--run-dir",
                        str(out / "processes" / name), "--wait-seconds", "3600", "--",
                        sys.executable, *command, "--config", str(out / "config.json")]
        record["command"] = full_command
        print(json.dumps({"stage": name, "status": "starting"}), flush=True)
        result = subprocess.run(full_command, cwd=ROOT)
        record.update(exit_code=result.returncode, finished_at=now())
        if result.returncode:
            manifest["status"] = "failed"
            save()
            raise SystemExit(result.returncode)
        save()
    summary = {"parameter_updates": 0, "selected_for_training": None, "students": {}}
    teacher = json.loads((out / "teacher/summary.json").read_text())
    summary["teacher_dev"] = teacher["splits"]["dev"]
    for spec in config["students"]:
        key = spec["key"]
        if key == "qwen_control":
            continue
        student = json.loads((out / key / "summary.json").read_text())
        probe = json.loads((out / f"{key}_probe/summary.json").read_text())
        summary["students"][key] = {
            "dev": student["splits"]["dev"], "probe_pairs": probe["pairs"],
            "teacher_minus_student_dev_accuracy": teacher["splits"]["dev"]["accuracy"] - student["splits"]["dev"]["accuracy"],
            "probe_summary": str((out / f"{key}_probe/summary.json").relative_to(ROOT)),
            "true_minus_random": probe["true_minus_random"],
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    manifest.update(status="complete", stage="steering_diagnosis_summary", finished_at=now())
    save()


if __name__ == "__main__":
    main()
