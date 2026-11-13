"""Paired fixed-prefix diagnostics only: no training or optimizer operations."""

import copy
import numpy as np
import torch

from steer_opd.direction_bank import bank_vectors, file_sha256
from steer_opd.resumable_core import atomic_json, atomic_torch, digest, response_hidden, score_hidden
from steer_opd.steering import CompositeSteeringController, SteeringController, find_decoder_layer


def position_metrics(student, teacher):
    if (student["ids"].shape != teacher["ids"].shape or student["logps"].shape != student["ids"].shape
            or teacher["logps"].shape != teacher["ids"].shape):
        raise ValueError("Overlap diagnostic needs identical token positions and K")
    matches = student["ids"][:, :, None] == teacher["ids"][:, None, :]
    sm, tm = matches.any(-1), matches.any(-2)
    return {"overlap": sm.float().mean(-1),
            "student_shared_mass": (student["logps"].exp() * sm).sum(-1),
            "teacher_shared_mass": (teacher["logps"].exp() * tm).sum(-1)}


def paired_summary(records, baseline, seed=42, bootstrap_samples=2000):
    ids = sorted(records)
    deltas = np.array([float(records[q]["overlap"].mean() - baseline[q]["overlap"].mean()) for q in ids])
    lengths = np.array([len(records[q]["overlap"]) for q in ids])
    candidate_sum = sum(float(records[q]["overlap"].sum()) for q in ids)
    baseline_sum = sum(float(baseline[q]["overlap"].sum()) for q in ids)
    rng = np.random.default_rng(seed)
    bootstrap = deltas[rng.integers(0, len(ids), size=(bootstrap_samples, len(ids)))].mean(-1)
    buckets = {}
    for label, lower, upper in [("0_512", 0, 512), ("513_2048", 513, 2048),
                                ("2049_4096", 2049, 4096), ("4097_plus", 4097, float("inf"))]:
        chosen = (lengths >= lower) & (lengths <= upper)
        if chosen.any():
            buckets[label] = {"questions": int(chosen.sum()), "question_weighted_delta_pp": float(deltas[chosen].mean() * 100)}
    return {"questions": len(ids), "positions": int(lengths.sum()), "length_buckets": buckets,
            "question_weighted_delta_pp": float(deltas.mean() * 100),
            "paired_bootstrap_95ci_pp": (np.quantile(bootstrap, [0.025, 0.975]) * 100).tolist(),
            "token_weighted_overlap16": candidate_sum / int(lengths.sum()),
            "token_weighted_delta_pp": 100 * (candidate_sum - baseline_sum) / int(lengths.sum()),
            "improved_questions": int((deltas > 0).sum()),
            "student_shared_mass": float(np.mean([float(records[q]["student_shared_mass"].mean()) for q in ids])),
            "teacher_shared_mass": float(np.mean([float(records[q]["teacher_shared_mass"].mean()) for q in ids])),
            "per_question_delta_pp": {qid: float(delta * 100) for qid, delta in zip(ids, deltas)},
            "parameter_updates": 0, "interpretation": "immediate_fixed_prefix_intervention_not_learning"}


def candidate_controller(model, bank, condition):
    if condition["kind"] == "baseline":
        return None
    if condition["kind"] == "legacy":
        payload = torch.load(condition["vector_path"], map_location="cpu", weights_only=True)
        return SteeringController(find_decoder_layer(model, payload["layer"]), payload["delta"], 1.0)
    selected = copy.copy(bank)
    selected["layers"] = {str(layer): bank["layers"][str(layer)] for layer in condition["layers"]}
    vectors, _ = bank_vectors(selected, rho=condition["rho"], control=condition.get("control", "positive"),
                              seed=condition.get("seed", 42))
    return CompositeSteeringController(model, vectors)


def evaluate_candidate(model, bank, condition, rows, teacher_dir, output, baseline_dir=None):
    controller = candidate_controller(model, bank, condition)
    results = {}
    try:
        with torch.inference_mode():
            for index, row in enumerate(rows):
                qid = row["id"]
                teacher_path = teacher_dir / f"{qid}.pt"
                binding = digest({"row": row, "condition": condition, "bank_id": bank["bank_id"],
                                  "teacher_sha256": file_sha256(teacher_path), "token_record_schema": 1})
                target = output / f"{qid}.pt"
                if target.exists():
                    record = torch.load(target, weights_only=True)
                    if record["binding"] != binding:
                        raise ValueError("Diagnostic prefix/vector/teacher changed")
                else:
                    teacher = torch.load(teacher_path, weights_only=True)
                    score = score_hidden(model, response_hidden(model, row, controller), include_entropy=True)
                    record = {"binding": binding, "ids": score["ids"], "logps": score["logps"],
                              **position_metrics(score, teacher), "parameter_updates": 0}
                    count = len(row["token_ids"])
                    record.update(token_record_schema=1, question_id=qid, bank_id=bank["bank_id"],
                        condition=condition, source_row_hash=digest(row),
                        response_token_ids=torch.tensor(row["token_ids"], dtype=torch.int32),
                        response_positions=torch.arange(count),
                        prediction_input_positions=torch.arange(len(row["prompt_token_ids"]) - 1,
                                                                len(row["prompt_token_ids"]) + count - 1),
                        steering_mask=torch.full((count,), controller is not None, dtype=torch.bool),
                        student_entropy=score["entropy"], entropy_support_size=score["entropy_support_size"],
                        teacher_record_path=str(teacher_path), teacher_record_sha256=file_sha256(teacher_path),
                        student_prefix_history="fixed_Base_tokens_with_this_condition_interventions",
                        current_step_probe=False)
                    if "entropy" in teacher:
                        record["teacher_entropy"] = teacher["entropy"]
                        record["signed_entropy_gap"] = score["entropy"] - teacher["entropy"]
                    else:
                        record["teacher_entropy_available"] = False
                    record["position_bins"] = {str(index): float(values.mean()) for index, values in
                                               enumerate(record["overlap"].tensor_split(min(10, len(record["overlap"]))))}
                    if baseline_dir is not None:
                        baseline = torch.load(baseline_dir / f"{qid}.pt", weights_only=True)
                        if condition.get("rho") == 0:
                            if not torch.equal(score["ids"], baseline["ids"]) or not torch.equal(score["logps"], baseline["logps"]):
                                raise RuntimeError("alpha0 is not exactly identical to baseline")
                        if index == 0:
                            restored = score_hidden(model, response_hidden(model, row))
                            if not torch.equal(restored["ids"], baseline["ids"]) or not torch.equal(restored["logps"], baseline["logps"]):
                                raise RuntimeError("Hook removal did not restore baseline")
                            record["hook_removal_exact"] = True
                    atomic_torch(target, record)
                results[qid] = record
    finally:
        if controller:
            controller.detach()
    atomic_json(output / "complete.json", {"questions": len(rows), "condition": condition, "parameter_updates": 0})
    return results
