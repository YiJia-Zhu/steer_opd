"""Frozen versioned student-space direction banks, shared by rollout and replay."""

import hashlib
import math

import torch

from steer_opd.resumable_core import digest


def file_sha256(path):
    sha = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def make_bank(pooled, method, identity, fixed_scales=None, *, allow_zero_directions=False):
    """Average raw differences before normalization; identical pair set per layer."""
    valid = {key: record for key, record in pooled.items() if record["usable"]}
    if not valid:
        raise ValueError("No usable direction pairs")
    layers = {}
    per_question = {key: {} for key in valid}
    for layer in method.layers:
        key = str(layer)
        differences = {qid: value["layers"][key]["pools"][0] - value["layers"][key]["pools"][1]
                       for qid, value in valid.items()}
        for difference in differences.values():
            if not torch.isfinite(difference).all() or (difference.norm() < 1e-8 and method.granularity == "global" and not allow_zero_directions):
                raise ValueError("Degenerate per-question direction; do not amplify noise")
        scale = (torch.as_tensor(fixed_scales[key]).float() if fixed_scales is not None else
                 torch.cat([n for value in valid.values() for n in value["layers"][key]["norms"]]).median())
        if not torch.isfinite(scale) or scale <= 0:
            raise ValueError("Invalid frozen activation scale")
        global_difference = torch.stack(list(differences.values())).mean(0)
        if global_difference.norm() < 1e-8 and method.granularity == "global" and not allow_zero_directions:
            raise ValueError("Degenerate global direction")
        layers[key] = {"direction": global_difference, "scale": scale}
        for qid, difference in differences.items():
            per_question[qid][key] = {"direction": difference, "scale": scale}
    bank = {"schema": 2, "method": method.to_dict(), "identity": identity, "layers": layers,
            "per_question": per_question, "valid_ids": list(valid),
            "invalid": {qid: r.get("reason", "unusable") for qid, r in pooled.items() if not r["usable"]}}
    bank["bank_id"] = digest({"method": bank["method"], "identity": identity, "valid_ids": list(valid),
                              "directions": {key: value["direction"].tolist() for key, value in layers.items()},
                              "per_question": {qid: {key: value["direction"].tolist() for key, value in item.items()}
                                               for qid, item in per_question.items()}})
    return bank


def bank_vectors(bank, question_id=None, *, rho=None, control=None, seed=None,
                 operator=None, alpha=None, layers=None):
    method = bank["method"]
    if method["granularity"] == "global":
        records, assigned = bank["layers"], "global"
    else:
        if question_id not in bank["per_question"]:
            raise ValueError(f"No usable vector for question {question_id}; no silent global fallback")
        assigned = question_id
        if method.get("vector_assignment", "own") == "other_question":
            ids = sorted(bank["per_question"])
            if len(ids) < 2:
                raise ValueError("Other-question control needs at least two questions")
            offset = 1 + method["seed"] % (len(ids) - 1)
            assigned = ids[(ids.index(question_id) + offset) % len(ids)]
        records = bank["per_question"][assigned]
    if layers is not None:
        if not layers or len(set(layers)) != len(layers) or any(str(layer) not in records for layer in layers):
            raise ValueError("Requested layers missing from direction bank")
        records = {str(layer): records[str(layer)] for layer in layers}
    operator = operator or method.get("operator", "additive")
    alpha = method.get("alpha") if alpha is None else alpha
    from steer_opd.operators import validate_operator
    strength = (rho if rho is not None else method["rho"]) if operator == "additive" else alpha if alpha is not None else float("nan")
    validate_operator(operator, strength)
    rho = method["rho"] if rho is None else rho
    control = control or method.get("direction_control", "positive")
    seed = method["seed"] if seed is None else seed
    if rho < 0 or not math.isfinite(rho) or control not in ("positive", "negative", "random"):
        raise ValueError("Invalid direction intervention")
    vectors = {}
    for key, value in records.items():
        direction = value["direction"].float()
        if direction.norm() < 1e-8:
            vectors[int(key)] = torch.zeros_like(direction)
            continue  # Zero difference means zero intervention, never a global fallback.
        if control == "random":
            question_seed = int(digest({"assigned_question": assigned})[:8], 16) if method["granularity"] == "per_question" else 0
            generator = torch.Generator().manual_seed(seed + int(key) + question_seed)
            direction = torch.randn(direction.shape, generator=generator)
        elif control == "negative":
            direction = -direction
        vectors[int(key)] = (direction / direction.norm() if operator == "slerp" else
                            (rho / math.sqrt(len(records))) * value["scale"] * direction / direction.norm())
    identity = {"bank_id": bank["bank_id"], "assigned": assigned, "rho": rho,
                "control": control, "seed": seed}
    if operator == "slerp":
        identity.update(operator=operator, alpha=alpha)
    if layers is not None:
        identity["selected_layers"] = list(layers)
    vector_id = digest(identity)
    return vectors, vector_id


def validate_request_vectors(path, requests):
    bank = torch.load(path, map_location="cpu", weights_only=True)
    sha = file_sha256(path)
    for request in requests:
        _, vector_id = bank_vectors(bank, request["id"])
        if request.get("vector_hash") != sha or request.get("vector_id") != vector_id:
            raise ValueError("Request vector file/hash/identity mismatch")
