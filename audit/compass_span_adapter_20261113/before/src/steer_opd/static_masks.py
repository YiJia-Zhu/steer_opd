"""Offline, immutable source-overlap masks and prediction-position mappings."""

import math

import torch

from steer_opd.resumable_core import digest


def source_binding(row, identity):
    if not row.get("id"):
        raise ValueError("Source needs a question ID")
    for key in ("prompt_token_ids", "token_ids"):
        ids = row.get(key)
        if not isinstance(ids, list) or not ids or any(type(t) is not int or t < 0 for t in ids):
            raise ValueError("Source needs nonempty, nonnegative integer prompt/response IDs")
    if row.get("steering") or any(row.get("steering_mask", [])):
        raise ValueError("A controlled trajectory cannot be a whole-source unsteered baseline")
    required = ("student_checkpoint", "teacher_checkpoint", "student_tokenizer_sha256",
                "teacher_tokenizer_sha256", "support_id", "metric", "student_score_sha256", "teacher_score_sha256")
    if any(not isinstance(identity.get(k), str) or not identity[k] for k in required):
        raise ValueError("Source mask needs checkpoint/tokenizer/support/metric and probability-file identities")
    if identity["metric"] == "native_overlap_at_16" and identity["student_tokenizer_sha256"] != identity["teacher_tokenizer_sha256"]:
        raise ValueError("Native token-ID overlap requires identical tokenizer mappings; align events explicitly")
    return digest({"id": row["id"], "key": row.get("key", row["id"]),
                   "prompt_token_ids": row["prompt_token_ids"], "token_ids": row["token_ids"], "identity": identity})


def freeze_static_mask(row, overlap, threshold, identity):
    """Freeze one strict-< comparison in FP64, before any candidate forward.

    Aligned cross-tokenizer metric values must be produced by their own
    accepted scorer; this function never reinterprets token IDs as aligned.
    """
    binding = source_binding(row, identity)
    if isinstance(threshold, bool) or not isinstance(threshold, (float, int)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Frozen threshold must be finite in [0,1]")
    values = torch.as_tensor(overlap, dtype=torch.float64, device="cpu").detach()
    if values.shape != (len(row["token_ids"]),) or not torch.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Source overlap needs one finite [0,1] value per response token")
    artifact = {"schema": 1, "mask_kind": "static_overlap", "source_binding": binding,
                "identity": dict(identity), "source_id": row["id"], "source_key": row.get("key", row["id"]),
                "source_overlap": values.tolist(), "threshold": float(threshold),
                "comparison": "strict_lt_float64", "requested_mask": (values < threshold).tolist(),
                "response_positions": list(range(len(values))),
                "prediction_input_positions": list(range(len(row["prompt_token_ids"])-1,
                                                          len(row["prompt_token_ids"])+len(values)-1)),
                "recompute_after_intervention": False}
    artifact["mask_id"] = digest(artifact)
    return artifact


def validate_static_mask(artifact, row, identity=None):
    identity = artifact["identity"] if identity is None else identity
    rebuilt = freeze_static_mask(row, artifact["source_overlap"], artifact["threshold"], identity)
    if artifact != rebuilt:
        raise ValueError("Static mask source/threshold/content/hash changed")
    return list(artifact["requested_mask"])


def selector_for_predictions(row, gates, *, mode):
    """Map prediction t to h[prompt_len-1+t], never to answer input t.

    Fixed-prefix teacher forcing puts those states in the engine prompt.
    A second generation uses last-prompt for t=0, decode step t-1 thereafter.
    Empty selections return None; adapters must disable steering explicitly.
    """
    if len(gates) != len(row["token_ids"]) or any(type(g) is not bool for g in gates):
        raise ValueError("Expected one boolean per source prediction")
    selected = [t for t, enabled in enumerate(gates) if enabled]
    if not selected:
        return None
    if mode == "fixed_prefix":
        return {"prompt_positions": [len(row["prompt_token_ids"])-1+t for t in selected]}
    if mode == "second_generation":
        clause = {}
        if 0 in selected:
            clause["prompt_positions"] = [-1]
        generation = [t-1 for t in selected if t > 0]
        if generation:
            clause["generation_positions"] = generation
        return clause
    raise ValueError("Unknown static-mask application stage")


def native_overlap(student_ids, teacher_ids, identity):
    if identity.get("metric") != "native_overlap_at_16" or identity.get("student_tokenizer_sha256") != identity.get("teacher_tokenizer_sha256"):
        raise ValueError("Cross-tokenizer IDs cannot be compared as native Overlap@16")
    a, b = torch.as_tensor(student_ids), torch.as_tensor(teacher_ids)
    if a.ndim != 2 or a.shape != b.shape or a.shape[1] != 16:
        raise ValueError("Native overlap requires paired [positions,16] arrays")
    if a.dtype.is_floating_point or b.dtype.is_floating_point or (a < 0).any() or (b < 0).any():
        raise ValueError("Top16 IDs must be nonnegative integers")
    if (a.sort(-1).values.diff(dim=-1) == 0).any() or (b.sort(-1).values.diff(dim=-1) == 0).any():
        raise ValueError("Duplicate Top16 IDs")
    return (a[:, :, None] == b[:, None, :]).any(-1).double().mean(-1)


def validate_native_score(score, length, support_size):
    ids, logps = score["ids"], score["logps"]
    if (ids.shape != (length, 16) or logps.shape != ids.shape or not 16 <= support_size
            or ids.dtype.is_floating_point or ids.dtype == torch.bool
            or (ids < 0).any() or (ids >= support_size).any()
            or not torch.isfinite(logps).all() or (logps > 1e-6).any()
            or (logps.diff(dim=-1) > 1e-6).any()
            or (logps.double().exp().sum(-1) > 1+1e-5).any()
            or (ids.sort(-1).values.diff(dim=-1) == 0).any()):
        raise ValueError("Invalid native Top16 probability cache")
