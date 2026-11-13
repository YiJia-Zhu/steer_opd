"""Reusable, versioned teacher token statistics for fixed-prefix diagnostics."""

import torch

from steer_opd.direction_bank import file_sha256
from steer_opd.resumable_core import atomic_json, atomic_torch, digest, response_hidden, score_hidden
from steer_opd.rollout_topk import tensor_hash


def prepare_teacher_statistics(rows, source, output, *, teacher_identity, support_size, loader, chunk=128):
    """Fill full-support entropy once; never infer it from normalized Top-K.

    Original fixed Top-K IDs/probabilities remain the diagnostic target. When
    old artifacts lack entropy, a same-checkpoint forward fills the statistic
    and verifies the old scores. Future candidates reuse this small cache.
    """
    model = None
    for row in rows:
        original_path = source / f"{row['id']}.pt"
        binding = digest({"schema": 1, "row": row, "teacher": teacher_identity,
                          "support_size": support_size, "source_sha256": file_sha256(original_path)})
        target = output / f"{row['id']}.pt"
        if target.exists():
            saved = torch.load(target, weights_only=True)
            if saved.get("statistics_binding") != binding:
                raise ValueError("Teacher token-statistics cache provenance changed")
            if saved.get("entropy_support_size") != support_size or saved["entropy"].shape != (len(row["token_ids"]),):
                raise ValueError("Teacher entropy support/position mismatch")
            continue
        original = torch.load(original_path, weights_only=True)
        if original["ids"].shape != (len(row["token_ids"]), 16):
            raise ValueError("Original teacher Top16 does not match fixed-prefix positions")
        if "entropy" in original and original.get("entropy_support_size") == support_size:
            entropy, error, origin = original["entropy"], 0.0, "existing_full_support_entropy"
        else:
            if model is None:
                model = loader()
            with torch.inference_mode():
                current = score_hidden(model, response_hidden(model, row), candidates=original["ids"],
                                       chunk=chunk, include_entropy=True)
            error = float((current["logps"] - original["logps"]).abs().max())
            if error > 5e-4 or current["entropy_support_size"] != support_size:
                raise ValueError(f"Same-checkpoint teacher cache changed: max logp error {error}")
            entropy, origin = current["entropy"], "same_checkpoint_full_support_forward"
        if entropy.shape != (len(row["token_ids"]),) or not torch.isfinite(entropy).all():
            raise ValueError("Invalid full-support teacher entropy")
        atomic_torch(target, {"ids": original["ids"], "logps": original["logps"], "entropy": entropy,
                    "entropy_support_size": support_size, "entropy_source": origin,
                    "cached_top16_replay_max_logp_error": error, "statistics_binding": binding,
                    "source_sha256": file_sha256(original_path), "question_id": row["id"],
                    "teacher_identity": teacher_identity})
    atomic_json(output / "complete.json", {"rows": len(list(output.glob("*.pt"))), "last_requested_rows": len(rows), "teacher": teacher_identity,
                "entropy_support_size": support_size, "parameter_updates": 0})


def save_trajectory_statistics(target, row, student, teacher_identity, support_size):
    """Keep per-position evidence after disposable training scores are cleaned."""
    count = len(row["token_ids"])
    ids, logps = student["teacher_topk_ids"], student["teacher_topk_logps"]
    matches = student["ids"][:, :, None] == ids[:, None, :]
    sm, tm = matches.any(-1), matches.any(-2)
    binding = digest({"schema": 2, "row_hash": digest(row), "distribution_hash": tensor_hash(student),
                      "teacher": teacher_identity, "support": support_size})
    if target.exists():
        if torch.load(target, weights_only=True).get("binding") != binding:
            raise ValueError("Committed token trajectory statistics changed")
        return
    record = {"schema": 2, "binding": binding, "row_hash": digest(row), "question_id": row.get("id", row.get("key")),
              "response_token_ids": torch.tensor(row["token_ids"], dtype=torch.int32),
              "response_positions": torch.arange(count),
              "prediction_input_positions": torch.arange(len(row["prompt_token_ids"]) - 1,
                                                          len(row["prompt_token_ids"]) + count - 1),
              "steering_mask": torch.tensor(row.get("steering_mask", [bool(row.get("steering"))] * count)),
              "vector_id": row.get("vector_id"), "vector_hash": row.get("vector_hash"),
              "vector_path": row.get("vector_path"), "support_size": support_size,
              "student_topk_file": row.get("rollout_topk"), "student_topk_hash": row.get("rollout_topk_hash"),
              "teacher_topk_ids": ids, "teacher_topk_logps": logps, "teacher_identity": teacher_identity,
              "teacher_on_student_logps": student["teacher_logps"],
              "intersection_count": sm.sum(-1).to(torch.uint8), "overlap16": sm.float().mean(-1),
              "student_shared_mass": (student["logps"].exp() * sm).sum(-1),
              "teacher_shared_mass": (logps.exp() * tm).sum(-1),
              "current_step_probe": "probe_topk_ids" in student,
              "prefix_source": "actual_rollout_controlled_history"}
    for key in ["student_entropy", "teacher_entropy", "probe_entropy"]:
        if key in student:
            record[key] = student[key]
    if "teacher_sampled_logps" in student:
        record["teacher_sampled_logps"] = student["teacher_sampled_logps"]
    record["entropy_available"] = {key: key in record for key in ["student_entropy", "teacher_entropy", "probe_entropy"]}
    if "student_entropy" in record and "teacher_entropy" in record:
        record["signed_entropy_gap"] = record["student_entropy"] - record["teacher_entropy"]
    if "probe_topk_ids" in student:
        probe_matches = student["probe_topk_ids"][:, :, None] == ids[:, None, :]
        record["probe_overlap16"] = probe_matches.any(-1).float().mean(-1)
        if "probe_entropy" in record and "teacher_entropy" in record:
            record["probe_signed_entropy_gap"] = record["probe_entropy"] - record["teacher_entropy"]
    atomic_torch(target, record)
