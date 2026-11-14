"""Batched fixed-prefix zero-update scoring with immutable source masks.

The teacher is a probability cache, never a callback in the student's loop.
The native metric adapter explicitly refuses mismatched tokenizer identities.
"""

from contextlib import nullcontext

import torch

from steer_opd.batched_controlled import BatchedController
from steer_opd.direction_bank import bank_vectors
from steer_opd.overlap_tuning import position_metrics
from steer_opd.resumable_core import digest, score_hidden
from steer_opd.static_masks import native_overlap, validate_static_mask


def candidate_vectors(bank, row, condition):
    if (bank["identity"].get("step") != 0
            or bank["method"]["direction"] != condition["direction"]
            or bank["method"]["granularity"] != condition["granularity"]):
        raise ValueError("Static candidate bank direction/granularity/S0 identity mismatch")
    return bank_vectors(bank, row["id"], operator=condition.get("operator", "slerp"),
                        alpha=condition.get("alpha", 0.1), rho=condition.get("rho"),
                        layers=condition["layers"])


def score_batch(model, rows, *, vectors=None, gates=None, operator="slerp", alpha=0.1,
                k=16, chunk=128, pad_token_id=0):
    """One causal model forward for a right-padded batch, with no probe/KV fork."""
    if not rows or (vectors is None) != (gates is None):
        raise ValueError("Need rows and either both vectors/masks or neither")
    if gates is not None and (len(rows) != len(gates) or len(rows) != len(vectors)):
        raise ValueError("One direction and mask per request required")
    lengths = [len(row["prompt_token_ids"])+len(row["token_ids"])-1 for row in rows]
    if any(not row["prompt_token_ids"] or not row["token_ids"] for row in rows):
        raise ValueError("Empty prompt/response")
    device = getattr(model, "_input_device", model.lm_head.weight.device)
    tokens = torch.full((len(rows), max(lengths)), pad_token_id, dtype=torch.long, device=device)
    attention = torch.zeros_like(tokens)
    mask = torch.zeros_like(tokens, dtype=torch.bool)
    for i, row in enumerate(rows):
        tokens[i, :lengths[i]] = torch.tensor(row["prompt_token_ids"]+row["token_ids"][:-1], device=device)
        attention[i, :lengths[i]] = 1
        if gates is not None:
            if len(gates[i]) != len(row["token_ids"]) or any(type(g) is not bool for g in gates[i]):
                raise ValueError("Static mask must contain one boolean per response token")
            mask[i, len(row["prompt_token_ids"])-1:lengths[i]] = torch.tensor(gates[i], device=device)
    controller = None
    if vectors is not None:
        layers = set(vectors[0])
        if not layers or any(set(v) != layers for v in vectors):
            raise ValueError("All rows in a scoring batch must target the same layer set")
        controller = BatchedController(model, {l: torch.stack([v[l] for v in vectors]) for l in layers},
                                       operator=operator, alpha=alpha, record_perturbations=True)
    results = []
    try:
        with torch.inference_mode(), controller.scope(mask) if controller else nullcontext():
            hidden = model.model(input_ids=tokens, attention_mask=attention, use_cache=False).last_hidden_state
            for i, row in enumerate(rows):
                start = len(row["prompt_token_ids"])-1
                scores = score_hidden(model, hidden[i, start:lengths[i]], k=k, chunk=chunk,
                                      sampled=row["token_ids"], include_entropy=True)
                scores["relative_perturbation"] = ({str(l): ratios[i, start:lengths[i]].cpu()
                    for l, ratios in controller.perturbations.items()} if controller else {})
                results.append(scores)
    finally:
        if controller:
            controller.detach()
    return results


def score_native_candidate(model, rows, teacher_records, masks, bank, condition, *,
                           baseline=None, batch_size=4, chunk=128, metric_adapter=None):
    """Native by default; cross-tokenizer scores need an explicit metric adapter."""
    if batch_size < 1 or len({row.get("key", row["id"]) for row in rows}) != len(rows):
        raise ValueError("Invalid batch size or duplicate response keys")
    prepared, vectors, metadata = [], [], []
    for row in rows:
        key = row.get("key", row["id"])
        artifact = masks[key]
        source_gates = validate_static_mask(artifact, row)
        identity = artifact["identity"]
        if identity["student_checkpoint"] != bank["identity"]["checkpoint"]:
            raise ValueError("Source and direction were extracted from different students")
        # This check precedes every model call, including the no-hook baseline.
        if metric_adapter is None:
            native_overlap(teacher_records[key]["ids"], teacher_records[key]["ids"], identity)
        elif identity["metric"] != metric_adapter.metric:
            raise ValueError("Candidate metric differs from the frozen source metric")
        if condition is None:
            values, vector_id, requested = None, None, [False]*len(source_gates)
        else:
            if condition["gate"] not in ("all", "static_overlap"):
                raise ValueError("Static scorer accepts all/static_overlap only")
            values, vector_id = candidate_vectors(bank, row, condition)
            requested = [True]*len(source_gates) if condition["gate"] == "all" else source_gates
        nonzero = values is not None and any(bool(torch.count_nonzero(v)) for v in values.values())
        enabled = nonzero and (condition.get("alpha", .1) != 0 if condition and condition.get("operator", "slerp") == "slerp" else True)
        actual = [bool(g and enabled) for g in requested]
        prepared.append(actual)
        vectors.append(values)
        metadata.append({"mask_id": artifact["mask_id"], "source_binding": artifact["source_binding"],
                         "source_overlap": torch.tensor([float("nan") if v is None else v for v in artifact["source_overlap"]], dtype=torch.float64),
                         "requested_mask": torch.tensor(requested), "steering_mask": torch.tensor(actual),
                         "zero_vector_no_intervention": bool(condition is not None and not nonzero),
                         "vector_id": vector_id, "condition": condition,
                         "binding": digest({"mask_id": artifact["mask_id"], "vector_id": vector_id,
                                            "condition": condition, "row": row})})
    results = {}
    for begin in range(0, len(rows), batch_size):
        subset = rows[begin:begin+batch_size]
        scores = score_batch(model, subset, vectors=vectors[begin:begin+batch_size] if condition else None,
            gates=prepared[begin:begin+batch_size] if condition else None,
            operator=condition.get("operator", "slerp") if condition else "slerp",
            alpha=condition.get("alpha", .1) if condition else .1, chunk=chunk)
        for j, (row, score) in enumerate(zip(subset, scores), begin):
            key = row.get("key", row["id"])
            metrics = (position_metrics(score, teacher_records[key]) if metric_adapter is None else
                       metric_adapter(key, score, teacher_records[key]))
            record = {**score, **metrics, **metadata[j],
                      "question_id": row["id"], "parameter_updates": 0, "current_step_probe": False,
                      "backend": "hf_batched_static_fixed_prefix",
                      "metric": "native_overlap_at_16" if metric_adapter is None else metric_adapter.metric}
            if baseline is not None:
                record["delta_overlap"] = record["overlap"]-baseline[key]["overlap"]
                if record.get("P_overlap_available"):
                    record["delta_P_overlap"] = record["P_overlap"]-baseline[key]["P_overlap"]
            results[key] = record
    return results
