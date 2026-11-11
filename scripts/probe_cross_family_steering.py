#!/usr/bin/env python3
"""Extract one student-space direction and probe cached ordinary prefixes.

No parameter updates and no direction selection by hooked overlap. Teacher and
student texts are replayed in the same S0, with matched content-token counts.
"""

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import time

import torch

from steer_opd.cross_family_overlap import compare_distributions
from steer_opd.math_data import read_rows
from steer_opd.runtime import verify_runtime
from steer_opd.steering import SteeringController, find_decoder_layer

from diagnose_cross_family_overlap import build_prompt, describe_model, load_model, score_state


ROOT = Path(__file__).resolve().parents[1]
METRICS = ["literal_continuation_overlap_at_16", "first_byte_overlap_lower_bound",
           "first_byte_overlap_upper_bound", "first_byte_jsd_with_residual_bucket_nats",
           "student_unresolved_probability_mass"]


def summarize(rows):
    return {key: sum(r[key] for r in rows) / len(rows) for key in METRICS}


def paired_change(rows, reference):
    """Cluster by question before resampling; prefix states are not independent."""
    groups = defaultdict(list)
    for row in rows:
        old = reference[row["state_id"]]
        groups[row["question_id"]].append({key: row[key] - old[key] for key in METRICS})
    per_question = [summarize(items) for items in groups.values()]
    generator = random.Random(42)
    samples = [{key: 0.0 for key in METRICS} for _ in range(2000)]
    for sample in samples:
        selected = generator.choices(per_question, k=len(per_question))
        sample.update(summarize(selected))
    result = {}
    for key in METRICS:
        values = sorted(s[key] for s in samples)
        result[key] = {"mean": sum(q[key] for q in per_question) / len(per_question),
                       "question_bootstrap_95pct": [values[49], values[1949]]}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--student", required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    runtime = verify_runtime()
    if not os.environ.get("STEER_OPD_GPU_ADMISSION"):
        raise RuntimeError("Launch through run_on_idle_gpus.py")
    out = args.output_dir.resolve()
    if not out.is_relative_to(ROOT) or out.exists():
        raise ValueError("Use a new project-local output directory")
    started = time.monotonic()
    config = json.loads(args.config.read_text())
    protocol = config["steering_probe"]
    spec = next(s for s in config["students"] if s["key"] == args.student)
    teacher_path = args.calibration_dir / "teacher/calibration.jsonl"
    student_path = args.calibration_dir / args.student / "calibration.jsonl"
    teachers = {r["id"]: r for r in read_rows(teacher_path)
                if r["correct"] and r["finish_reason"] != "length"}
    students = read_rows(student_path)
    pairs = [(teachers[r["id"]], r) for r in students if r["id"] in teachers][:protocol["pair_limit"]]
    states = read_rows(args.baseline_dir / args.student / "states.jsonl")
    if {s["question_id"] for s in states} & {s["id"] for _, s in pairs}:
        raise ValueError("Calibration/diagnosis leakage")
    if len(pairs) < protocol["minimum_pairs"]:
        raise ValueError("Insufficient independent teacher-validated calibration pairs")
    tokenizer, model = load_model(spec)
    info = describe_model(tokenizer, model, spec)
    layer_index = int(model.config.num_hidden_layers * protocol["relative_layer_depth"])
    layer = find_decoder_layer(model, layer_index)
    excluded = set(tokenizer.all_special_ids)
    captured = []
    handle = layer.register_forward_hook(
        lambda module, inputs, output: captured.append(output[0] if isinstance(output, tuple) else output))
    deltas, norms, lengths = [], [], []
    try:
        with torch.inference_mode():
            for teacher, student in pairs:
                if teacher["question"] != student["question"]:
                    raise ValueError("Mismatched calibration questions")
                prompt = build_prompt(tokenizer, student["question"], spec)
                # Both sources are text, tokenized afresh by the same frozen S0.
                encoded = [tokenizer.encode(r["text"], add_special_tokens=False)
                           for r in (teacher, student)]
                content_ids = [[i for i, value in enumerate(ids) if value not in excluded] for ids in encoded]
                count = min(protocol["content_token_cap"], *(len(ids) for ids in content_ids))
                if count == 0:
                    raise ValueError("Empty calibration response")
                pools = []
                for ids, positions in zip(encoded, content_ids):
                    positions = positions[:count]
                    ids = ids[:positions[-1] + 1]
                    captured.clear()
                    tokens = torch.tensor([prompt + ids], device="cuda")
                    model.model(input_ids=tokens, use_cache=False)
                    content = captured[0][0, [len(prompt) + p for p in positions], :].float()
                    pools.append(content.mean(0).cpu())
                    norms.append(content.norm(dim=-1).cpu())
                lengths.append({"id": student["id"], "matched_content_tokens": count,
                                "teacher_content_tokens": len(content_ids[0]),
                                "student_content_tokens": len(content_ids[1]),
                                "student_truncated": student["finish_reason"] == "length"})
                deltas.append(pools[0] - pools[1])
                print(json.dumps({"student": args.student, "extracted_pairs": len(deltas),
                                  "total": len(pairs)}), flush=True)
    finally:
        handle.remove()
        captured.clear()
    direction = torch.stack(deltas).mean(0)
    median_norm = torch.cat(norms).median().item()
    if not torch.isfinite(direction).all() or not direction.norm().item():
        raise ValueError("Invalid direction")
    random_vector = torch.randn(direction.shape, generator=torch.Generator().manual_seed(config["seed"]))
    random_vector *= direction.norm() / random_vector.norm()
    out.mkdir(parents=True)
    torch.save({"direction": direction, "random": random_vector}, out / "vectors.pt")
    teacher_scores = {r["state_id"]: r["distribution"]
                      for r in read_rows(args.baseline_dir / "teacher_scores.jsonl")}
    baseline = {r["state_id"]: r for r in read_rows(args.baseline_dir / "comparisons.jsonl")
                if r["student"] == args.student}
    rollouts = {r["id"]: r for r in read_rows(args.baseline_dir / args.student / "rollouts.jsonl")}
    # Check replay and alpha=0 before interpreting any intervention.
    checks = []
    for state in states[:3]:
        current = score_state(tokenizer, model, state["student_input_ids"], info,
                              config["native_probability_topk"])
        result = compare_distributions(teacher_scores[state["state_id"]], current, config["overlap_k"])
        difference = max(abs(result[k] - baseline[state["state_id"]][k]) for k in METRICS)
        if difference > 1e-6:
            raise ValueError(f"Unsteered replay differs from baseline: {difference}")
        checks.append({"state_id": state["state_id"], "baseline_metric_max_abs_difference": difference})
    first = states[0]
    zero = SteeringController(layer, direction, 0)
    try:
        with zero.loss_scope(torch.ones(1, len(first["student_input_ids"]), dtype=torch.bool)):
            zero_distribution = score_state(tokenizer, model, first["student_input_ids"], info,
                                            config["native_probability_topk"])
        if zero_distribution != first["student_distribution"]:
            raise ValueError("Zero-strength hook changed student scores")
    finally:
        zero.detach()
    variants, all_results = {}, {}
    with (out / "comparisons.jsonl").open("x") as stream:
        for rho in protocol["rhos"]:
            alpha = rho * median_norm / direction.norm().item()
            for label, vector in [("true", direction), ("random", random_vector), ("negative", -direction)]:
                key = f"{label}_rho{rho}"
                results = []
                controller = SteeringController(layer, vector, alpha)
                try:
                    for state in states:
                        ids = state["student_input_ids"]
                        prompt_length = len(rollouts[state["question_id"]]["prompt_token_ids"])
                        mask = torch.zeros((1, len(ids)), dtype=torch.bool)
                        # Include the prompt's last state, which predicts response token 0.
                        mask[:, prompt_length - 1:] = True
                        with controller.loss_scope(mask):
                            distribution = score_state(tokenizer, model, ids, info,
                                                       config["native_probability_topk"])
                        result = {"state_id": state["state_id"], "question_id": state["question_id"],
                                  "prefix_position": state["prefix_position"], "variant": key,
                                  **compare_distributions(teacher_scores[state["state_id"]], distribution,
                                                          config["overlap_k"])}
                        results.append(result)
                        stream.write(json.dumps(result) + "\n")
                finally:
                    controller.detach()
                variants[key] = {"direction": label, "rho": rho, "alpha": alpha,
                                 "states": len(results), "means": summarize(results),
                                 "paired_change_from_baseline": paired_change(results, baseline),
                                 "by_prefix_position": {
                                     str(pos): summarize([r for r in results if r["prefix_position"] == pos])
                                     for pos in config["prefix_token_positions"]}}
                all_results[key] = results
                stream.flush()
                print(json.dumps({"student": args.student, "variant": key, "states": len(results)}), flush=True)
    restored = score_state(tokenizer, model, first["student_input_ids"], info, config["native_probability_topk"])
    if restored != first["student_distribution"]:
        raise ValueError("Removing hooks did not restore baseline")
    true_random = {}
    for rho in protocol["rhos"]:
        reference = {r["state_id"]: r for r in all_results[f"random_rho{rho}"]}
        true_random[str(rho)] = paired_change(all_results[f"true_rho{rho}"], reference)
    summary = {"spec": spec, "runtime": runtime, "layer": layer_index,
               "relative_layer_depth": protocol["relative_layer_depth"], "pairs": len(pairs),
               "median_activation_norm": median_norm, "direction_norm": direction.norm().item(),
               "random_norm": random_vector.norm().item(), "pooling": "Per-answer mean, equal per-pair weight; matched content lengths",
               "lengths": lengths, "teacher_answers_sha256": hashlib.sha256(teacher_path.read_bytes()).hexdigest(),
               "student_answers_sha256": hashlib.sha256(student_path.read_bytes()).hexdigest(),
               "fixed_baseline_dir": str(args.baseline_dir.resolve()), "variants": variants,
               "true_minus_random": true_random, "replay_checks": checks,
               "alpha_zero_unchanged": True, "hook_removal_restored_baseline": True,
               "parameter_updates": 0, "selected_for_training": None,
               "elapsed_seconds": time.monotonic() - started,
               "gpu_admission": os.environ["STEER_OPD_GPU_ADMISSION"],
               "limitations": ["Small calibration/dev pilot; one random direction seed.",
                               "Hooked distribution changes do not establish reasoning compatibility or learning gains.",
                               "First-byte agreement is coarse; literal agreement depends on segmentation.",
                               "Direction and layer were not selected using task learning gains."]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
