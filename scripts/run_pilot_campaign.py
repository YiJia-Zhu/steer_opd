#!/usr/bin/env python3
"""Bounded first campaign: initial MATH-500, calibration, then three 100-step runs.

Each phase separately checks idle GPUs. A failed phase stops the queue, without
killing other processes. Completed phases are never overwritten or retrained.
"""

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from steer_opd.runtime import verify_runtime


ROOT = Path(__file__).resolve().parents[1]
STUDENT = "/home/zhenning/data/huggingface_models/DeepSeek-R1-Distill-Qwen-1.5B"
TEACHER = "/home/zhenning/data/huggingface_datasets/JustRL-DeepSeek-1.5B"
DATA = ROOT / "data/math500_pilot_v2"


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _resolve_recorded_model_path(path):
    """Keep historical evaluations reusable after the dataset-directory rename."""
    recorded = Path(path)
    for parent in [Path("/home/zhenning/data"), Path("/local_nvme/data/zhenning")]:
        legacy_root = parent / "hugggingface_datasets"
        if recorded.is_relative_to(legacy_root):
            return (parent / "huggingface_datasets" / recorded.relative_to(legacy_root)).resolve()
    return recorded.resolve()


def verify_initial_evaluation(path):
    path = path.resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Reused results must stay inside this project")
    process = json.loads((path / "process.json").read_text())
    if process.get("exit_code") != 0:
        raise ValueError("Cannot reuse an incomplete initial evaluation")
    input_hash = hashlib.sha256((DATA / "math500.jsonl").read_bytes()).hexdigest()
    records = {}
    for role, model in [("student", STUDENT), ("teacher", TEACHER)]:
        summary = json.loads((path / f"{role}.summary.json").read_text())
        expected = {"samples": 2000, "questions": 500, "n": 4, "seed": 42,
                    "max_tokens": 31744, "temperature": 0.7, "top_p": 0.95,
                    "top_k": -1, "steering": False, "input_sha256": input_hash}
        if any(summary.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Reused {role} evaluation protocol differs")
        if _resolve_recorded_model_path(summary["model"]) != Path(model).resolve():
            raise ValueError(f"Reused {role} model path differs")
        if hashlib.sha256((path / f"{role}.jsonl").read_bytes()).hexdigest() != summary["output_sha256"]:
            raise ValueError(f"Reused {role} output hash differs")
        records[role] = {"summary": str(path / f"{role}.summary.json"),
                         "output_sha256": summary["output_sha256"], "avg_accuracy": summary["avg_accuracy"]}
    fingerprints = json.loads((ROOT / "audit/math500_launch_2026-11-08/weight_fingerprints.json").read_text())
    for key, directory in [("student_r1_1p5b", STUDENT), ("teacher_justrl_1p5b", TEACHER)]:
        for item in fingerprints["models"][key]:
            digest = hashlib.sha256()
            with (Path(directory) / item["file"]).open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != item["sha256"]:
                raise ValueError(f"Model weights changed since initial evaluation: {key}")
    return {"directory": str(path), "results": records, "model_weight_hashes_verified": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reuse-initial-math500", type=Path,
                        help="Reuse a completed, hash-verified initial evaluation; continue at dev")
    args = parser.parse_args()
    runtime = verify_runtime()
    reused_initial = verify_initial_evaluation(args.reuse_initial_math500) if args.reuse_initial_math500 else None
    for smoke in ["e0_smoke_02", "e3_smoke_01"]:
        if json.loads((ROOT / "runs" / smoke / "process.json").read_text()).get("exit_code") != 0:
            raise RuntimeError(f"Integration prerequisite has not passed: {smoke}")
    split_manifest = json.loads((DATA / "split_manifest.json").read_text())
    for filename, info in split_manifest["files"].items():
        if hashlib.sha256((DATA / filename).read_bytes()).hexdigest() != info["sha256"]:
            raise ValueError(f"Prepared dataset changed: {filename}")
    out = args.output_dir.resolve()
    if not out.is_relative_to(ROOT) or (out / "campaign.json").exists():
        raise ValueError("Use a new project-local campaign directory")
    out.mkdir(parents=True, exist_ok=True)
    if reused_initial:
        (out / "initial_math500").symlink_to(reused_initial["directory"], target_is_directory=True)
    vector_dir = out / "vectors"
    teacher_answers = out / "calibration_teacher/answers.jsonl"
    student_answers = out / "calibration_student/answers.jsonl"
    stages = [
        ("initial_math500", 2, ["scripts/evaluate_initial_pair.py", "--output-dir", str(out / "initial_math500")]),
        ("initial_dev", 2, ["scripts/evaluate_initial_pair.py", "--output-dir", str(out / "initial_dev"),
         "--input", str(DATA / "dev_pilot.jsonl"), "--n", "1", "--max-tokens", "7168"]),
        ("calibration_teacher", 1, ["scripts/generate_math.py", "--model", TEACHER,
         "--input", str(DATA / "calibration_candidates.jsonl"), "--output", str(teacher_answers),
         "--limit", "512", "--n", "1", "--max-tokens", "7168", "--correct-target", "128", "--skip-overlong"]),
        ("calibration_student", 1, ["scripts/generate_math.py", "--model", STUDENT,
         "--input", str(DATA / "calibration_candidates.jsonl"), "--output", str(student_answers),
         "--correct-ids-from", str(teacher_answers), "--limit", "128", "--n", "1", "--max-tokens", "7168"]),
        ("extract_vectors", 1, ["scripts/extract_pilot_vectors.py", "--teacher-answers", str(teacher_answers),
         "--student-answers", str(student_answers), "--output-dir", str(vector_dir)]),
    ]
    for group in ["E0", "E3", "E4"]:
        command = ["scripts/launch_pilot.py", "--group", group, "--run-dir", str(out / group),
                   "--steps", "100", "--batch-size", "16", "--response-length", "2048"]
        if group != "E0":
            command += ["--steering-config", str(vector_dir / f"{group}.json")]
        stages.append((group, 2, command))
    manifest = {"started_at": now(), "pid": os.getpid(), "status": "running",
                "objective": "Single-seed short-sequence engineering pilot, not formal method evidence",
                "training_updates_per_group": 100, "seed": 42,
                "runtime": runtime, "reused_initial_evaluation": reused_initial,
                "current_stage": None, "stages": [],
                "data_manifest_sha256": hashlib.sha256((DATA / "split_manifest.json").read_bytes()).hexdigest(),
                "source_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for folder in [ROOT / "src/steer_opd", ROOT / "scripts"]
                                  for p in sorted(folder.glob("*.py"))}}
    status = out / "campaign.json"
    for name, count, arguments in stages:
        stage_dir = out / name
        if name == "initial_math500" and reused_initial:
            manifest["stages"].append({"name": name, "status": "reused", "exit_code": 0,
                                       "source_directory": reused_initial["directory"], "verified_at": now()})
            status.write_text(json.dumps(manifest, indent=2) + "\n")
            continue
        command = [sys.executable, "scripts/run_on_idle_gpus.py", "--count", str(count),
                   "--run-dir", str(stage_dir), "--wait-seconds", "3600", "--", sys.executable, *arguments]
        record = {"name": name, "started_at": now(), "gpus_requested": count, "command": command}
        manifest["current_stage"] = name
        manifest["stages"].append(record)
        status.write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps({"stage": name, "status": "starting", "time": now()}), flush=True)
        try:
            record["runtime"] = verify_runtime()
            result = subprocess.run(command, cwd=ROOT)
        except Exception as error:
            record.update({"exception": repr(error), "finished_at": now()})
            manifest["status"] = "failed"
            status.write_text(json.dumps(manifest, indent=2) + "\n")
            raise
        record.update({"exit_code": result.returncode, "finished_at": now()})
        if result.returncode:
            manifest["status"] = "failed"
            status.write_text(json.dumps(manifest, indent=2) + "\n")
            raise SystemExit(result.returncode)
        if name == "initial_dev":
            summaries = {role: json.loads((stage_dir / f"{role}.summary.json").read_text())
                         for role in ["student", "teacher"]}
            manifest["teacher_dev_advantage"] = summaries["teacher"]["avg_accuracy"] - summaries["student"]["avg_accuracy"]
            manifest["formal_scaling_allowed_by_dev_advantage"] = manifest["teacher_dev_advantage"] > 0
        status.write_text(json.dumps(manifest, indent=2) + "\n")
    # Baseline dev and post-training evaluation are deliberately explicit phases;
    # no MATH-500 score is used to select vectors or checkpoints.
    eval_models = []
    for group in ["E0", "E3", "E4"]:
        model = out / group / "checkpoints/global_step_100/actor/huggingface"
        if not (model / "config.json").exists():
            manifest["status"] = "failed_missing_hf_checkpoint"
            status.write_text(json.dumps(manifest, indent=2) + "\n")
            raise FileNotFoundError(model)
        eval_models.append((group, str(model)))
    eval_specs = [(name + "_dev", model, "dev_pilot", 1, 7168) for name, model in eval_models]
    eval_specs += [(name + "_math500", model, "math500", 4, 31744)
                   for name, model in eval_models if name in {"E0", "E3", "E4"}]
    for name, model, dataset, n, max_tokens in eval_specs:
        stage_dir = out / name
        command = [sys.executable, "scripts/run_on_idle_gpus.py", "--count", "1",
                   "--run-dir", str(stage_dir), "--wait-seconds", "3600", "--", sys.executable,
                   "scripts/generate_math.py", "--model", model, "--input", str(DATA / f"{dataset}.jsonl"),
                   "--output", str(stage_dir / "answers.jsonl"), "--n", str(n), "--max-tokens", str(max_tokens)]
        record = {"name": name, "started_at": now(), "gpus_requested": 1, "command": command}
        manifest["current_stage"] = name
        manifest["stages"].append(record)
        status.write_text(json.dumps(manifest, indent=2) + "\n")
        try:
            record["runtime"] = verify_runtime()
            result = subprocess.run(command, cwd=ROOT)
        except Exception as error:
            record.update({"exception": repr(error), "finished_at": now()})
            manifest["status"] = "failed"
            status.write_text(json.dumps(manifest, indent=2) + "\n")
            raise
        record.update({"exit_code": result.returncode, "finished_at": now()})
        if result.returncode:
            manifest["status"] = "failed"
            status.write_text(json.dumps(manifest, indent=2) + "\n")
            raise SystemExit(result.returncode)
    manifest.update({"status": "complete", "finished_at": now(), "current_stage": None})
    status.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "complete", "path": str(status)}), flush=True)


if __name__ == "__main__":
    main()
