#!/usr/bin/env python3
"""Create a paused, independent method variant. Never launch GPU work."""

import argparse
import re

from steer_opd.method_config import resolve_method
from steer_opd.resumable_core import ROOT, atomic_json, read_json


def build_variant(base, preset, name, **overrides):
    if not re.fullmatch(r"[a-z0-9_]+", name):
        raise ValueError("Variant name must use lowercase letters, digits and underscores")
    method = {**preset, **{key: value for key, value in overrides.items() if value is not None and key != "reference"}}
    cfg = {**base, "id": name, "run_dir": f"runs/{name}", "data_dir": f"data/{name}",
           "reuse_question_splits_from": base["data_dir"], "direction_data_policy": "same_as_opd",
           "direction_questions": base["train_questions"], "authorized": False,
           "execution_hold": "user_must_resume_after_review_and_gpu_acceptance", "steering_method": method,
           "direction": method["direction"], "student_topk_source": "rollout"}
    if overrides.get("reference"):
        cfg["reference"] = overrides["reference"]
    resolved = resolve_method(cfg)
    cfg["steering_method"] = resolved.to_dict()
    cfg["rho"] = resolved.rho
    if len(resolved.layers) == 1:
        cfg["layer"] = resolved.layers[0]
    cfg["injection"] = "all_response_prediction_states_rollout_and_replay" if resolved.gate == "all" else "recorded_causal_gate_mask_rollout_and_replay"
    cfg["deviations"] = ["100 shared OT3 calibration/OPD questions; independent original dev/AIME",
                         "Optional method branch; fresh Base initialization; no SFT",
                         "GPU validation and actual cost measurement pending; no experiment authorized by config creation"]
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="configs/figure8_shared_top16_v2.json")
    parser.add_argument("--presets", default="configs/steering_branch_presets.json")
    parser.add_argument("--preset", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--reference")
    parser.add_argument("--layers", nargs="+", type=int)
    parser.add_argument("--rho", type=float)
    parser.add_argument("--overlap-threshold", type=float)
    parser.add_argument("--entropy-threshold", type=float)
    parser.add_argument("--random-probability", type=float)
    parser.add_argument("--start-token", type=int)
    parser.add_argument("--direction-control", choices=["positive", "negative", "random"])
    parser.add_argument("--vector-assignment", choices=["own", "other_question"])
    parser.add_argument("--refresh-interval", type=int)
    parser.add_argument("--matching-bin-width", type=int)
    args = parser.parse_args()
    presets = read_json(args.presets)["presets"]
    cfg = build_variant(read_json(args.base), presets[args.preset], args.name,
            reference=args.reference, overlap_threshold=args.overlap_threshold, entropy_threshold=args.entropy_threshold,
            layers=args.layers, rho=args.rho,
            random_probability=args.random_probability, start_token=args.start_token,
            direction_control=args.direction_control, vector_assignment=args.vector_assignment,
            refresh_interval=args.refresh_interval, matching_bin_width=args.matching_bin_width)
    path = ROOT / "configs" / f"{args.name}.json"
    if path.exists():
        raise ValueError("Use a new variant name; existing scientific configuration is immutable")
    atomic_json(path, cfg)
    print(f"Created paused variant: {path}; no GPU/data generation started")


if __name__ == "__main__":
    main()
