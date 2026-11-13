#!/usr/bin/env python3
"""Score frozen S0 with/without directions; always exits without training."""

import argparse
import os
import torch

from steer_opd.overlap_tuning import evaluate_candidate, paired_summary
from steer_opd.resumable_core import ROOT, atomic_json, load_model, read_json
from steer_opd.method_config import resolve_method
from steer_opd.token_statistics import prepare_teacher_statistics


def score_job(job):
    cfg = read_json(ROOT / job["config"])
    method = resolve_method(cfg)
    if (cfg["optimizer_updates"] != 0 or method.direction != "T-S" or method.granularity != "global"
            or method.extractor != "initial" or method.gate != "all"):
        raise ValueError("First diagnostic is frozen T-S with zero parameter updates")
    os.environ["STEER_OPD_SUPPORT_SIZE"] = str(cfg["shared_vocabulary_size"])
    bank = torch.load(ROOT / job["bank"], weights_only=True)
    if (bank["identity"]["step"] != 0 or bank["identity"]["checkpoint"] != cfg["student"]
            or any(bank["method"][key] != getattr(method, key) for key in ["direction", "granularity", "extractor", "gate"])):
        raise ValueError("First diagnostic requires an initial-student global bank")
    rows = read_json(ROOT / job["rows"])
    root = ROOT / job["output"]
    teacher = ROOT / cfg["run_dir"] / "teacher_token_statistics"
    prepare_teacher_statistics(rows, ROOT / job["teacher_scores"], teacher,
        teacher_identity=cfg["teacher"], support_size=cfg["shared_vocabulary_size"],
        loader=lambda: load_model(ROOT / cfg["teacher"]), chunk=cfg["logit_chunk_size"])
    model = load_model(ROOT / cfg["student"])
    baseline_condition = {"id": "baseline", "kind": "baseline"}
    baseline = evaluate_candidate(model, bank, baseline_condition, rows, teacher, root / "baseline")
    summaries = {}
    for condition in job["conditions"]:
        condition = dict(condition)
        if condition["kind"] == "legacy":
            condition["vector_path"] = str(ROOT / condition["vector_path"])
        records = evaluate_candidate(model, bank, condition, rows, teacher, root / condition["id"], root / "baseline")
        summaries[condition["id"]] = {"condition": condition, **paired_summary(records, baseline, cfg["seed"])}
        atomic_json(root / "summaries.json", summaries)
    atomic_json(root / "complete.json", {"parameter_updates": 0, "candidates": len(summaries),
                "next_action": "stop_for_user_review", "automatic_opd": False})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    score_job(read_json(args.job))


if __name__ == "__main__":
    main()
