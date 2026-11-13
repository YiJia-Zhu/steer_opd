#!/usr/bin/env python3
"""Read-only progress view; no CUDA initialization and no process signals."""

import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from steer_opd.campaign_status import supervisor_is_alive


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("runs/figure8_first_run_20261112"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run = args.run_dir.resolve()
    status = read(run / "campaign.json")
    status["supervisor_alive"] = supervisor_is_alive(status)
    if status.get("status") == "running" and not status["supervisor_alive"]:
        status["recorded_status"] = "running"
        status["status"] = "interrupted_resumable"
        status["status_reason"] = "Recorded supervisor is absent or its PID belongs to a different process"
    status["groups"] = {}
    for group in ["vanilla", "steer"]:
        metrics = [read(p) for p in (run / group / "steps").glob("*/train_metrics.json")]
        latest = max([read(run / group / "latest.json").get("step", 0), *[r["step"] for r in metrics]])
        pending = run / group / "steps" / f"{latest + 1:04d}"
        status["groups"][group] = {"committed_updates": latest,
            "pending_step": latest + 1, "rollouts_saved": len(list((pending / "rollouts").glob("[0-9]*.json"))),
            "rollout_topk_saved": len(list((pending / "rollouts/topk").glob("*.pt"))),
            "student_scores_saved": len(list((pending / "student_scores").glob("*.pt"))),
            "teacher_scores_saved": len(list((pending / "scores").glob("*.pt")))}
    if status.get("attempt"):
        process = read(root / status["attempt"] / "process.json")
        status["gpu_process"] = {k: process.get(k) for k in ["pid", "gpus", "exit_code"]}
        log = root / status["attempt"] / "stdout.log"
        if log.exists():
            with log.open("rb") as stream:
                stream.seek(max(0, log.stat().st_size - 4000))
                status["recent_log"] = stream.read().decode(errors="replace").splitlines()[-5:]
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
