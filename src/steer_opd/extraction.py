"""Read text pairs in one frozen student checkpoint, without a training graph."""

import re

import torch

from steer_opd.resumable_core import atomic_torch, digest, emit, read_json
from steer_opd.direction_bank import file_sha256


def encode_content(tokenizer, text, with_offsets=False):
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    structural = [m.span() for m in re.finditer(r"</?(?:think|tool_response)>|<\|[^>]+\|>", text)]
    excluded = set(tokenizer.all_special_ids)
    mask = [token not in excluded and b > a and not any(a < end and b > begin for begin, end in structural)
            for token, (a, b) in zip(encoded["input_ids"], encoded["offset_mapping"])]
    if with_offsets:
        return encoded["input_ids"], mask, encoded["offset_mapping"]
    return encoded["input_ids"], mask


def pool_question(model, tokenizer, item, records, layers, token_capture=None, cache_dtype=torch.bfloat16, allow_zero=False):
    if any(r["finish_reason"] == "length" or not r["text"].strip() for r in records):
        return {"usable": False, "reason": "empty_or_truncated"}
    for row in records:
        words = row["text"].split()
        if len(words) > 128 and len(set(words[-128:])) < 8:
            return {"usable": False, "reason": "severe_exact_repetition"}
    captured = {}
    handles = []
    for layer in layers:
        def capture(module, args, output, index=layer):
            captured[index] = output[0] if isinstance(output, tuple) else output
        handles.append(model.model.layers[layer].register_forward_hook(capture))
    result = {str(layer): {"pools": [], "norms": []} for layer in layers}
    device = model.lm_head.weight.device
    try:
        for record_index, row in enumerate(records):
            response, mask, offsets = encode_content(tokenizer, row["text"], with_offsets=True)
            if not any(mask):
                return {"usable": False, "reason": "no_answer_content"}
            prefix = item["student_prompt_ids"]
            tokens = torch.tensor([prefix + response], device=device)
            captured.clear()
            with torch.inference_mode():
                model.model(input_ids=tokens, use_cache=False)
            if token_capture is not None:
                # Row0 is the last prompt state: it predicts the first answer
                # token. Rows1..L are the observed answer states used for Phi.
                token_capture(record_index, {
                    "student_prompt_ids": torch.tensor(prefix, dtype=torch.int32),
                    "student_response_ids": torch.tensor(response, dtype=torch.int32),
                    "response_offsets": torch.tensor(offsets, dtype=torch.int32),
                    "offset_semantics": "student_tokenizer_offset_mapping",
                    "body_mask": torch.tensor(mask, dtype=torch.bool),
                    "state_input_positions": torch.arange(len(prefix) - 1, len(prefix) + len(response)),
                    "prediction_state_rows": torch.arange(len(response)),
                    "response_token_state_rows": torch.arange(1, len(response) + 1),
                    "states": {str(layer): captured[layer][0, len(prefix) - 1:].detach().to(
                        device="cpu", dtype=cache_dtype).contiguous() for layer in layers},
                    "source_dtypes": {str(layer): str(captured[layer].dtype) for layer in layers},
                    "cache_dtype": str(cache_dtype), "steering": False})
            for layer in layers:
                states = captured[layer][0, len(prefix):].float()[torch.tensor(mask, device=device)]
                result[str(layer)]["pools"].append(states.mean(0).cpu())
                result[str(layer)]["norms"].append(states.norm(dim=-1).cpu())
        if len(records) == 2:
            for record in result.values():
                difference = record["pools"][0] - record["pools"][1]
                if not torch.isfinite(difference).all() or (difference.norm() < 1e-8 and not allow_zero):
                    return {"usable": False, "reason": "degenerate_difference_at_candidate_layer"}
        return {"usable": True, "layers": result}
    finally:
        for handle in handles:
            handle.remove()


def extract_pairs(model, tokenizer, questions, role_directories, method, output, identity,
                  cache_token_activations=False, cache_dtype="bfloat16"):
    if cache_dtype not in ("bfloat16", "float32"):
        raise ValueError("Token activation cache dtype must be bfloat16 or float32")
    pooled = {}
    for index, question in enumerate(questions):
        records = [read_json(role_directories[role] / f"{question['id']}.json") for role in method.roles]
        for record in records:
            if record["id"] != question["id"]:
                raise ValueError("Calibration answer belongs to another question")
        pair_identity = {"question": question, "records": records, "identity": identity,
                         "layers": method.layers, "roles": method.roles}
        if cache_token_activations:
            pair_identity["token_cache"] = {"schema": 1, "dtype": cache_dtype}
        pair_hash = digest(pair_identity)
        target = output / "pooled" / f"{question['id']}.pt"
        artifacts = {}
        def save_states(record_index, payload):
            role = method.roles[record_index]
            path = output / "token_activations" / question["id"] / f"{role}.pt"
            atomic_torch(path, {"schema": 1, "question_id": question["id"], "role": role,
                               "pair_hash": pair_hash, "identity": identity,
                               "reference_hash": digest(records[record_index]), **payload})
            artifacts[role] = {"path": str(path.relative_to(output)), "sha256": file_sha256(path)}
        if target.exists():
            value = torch.load(target, weights_only=True)
            if value["pair_hash"] != pair_hash:
                raise ValueError("Calibration texts/checkpoint changed during extraction")
            if cache_token_activations:
                validate_token_artifacts(value, output, method.roles)
        else:
            value = {**pool_question(model, tokenizer, question, records, method.layers,
                                    save_states if cache_token_activations else None,
                                    getattr(torch, cache_dtype), allow_zero=method.granularity == "per_question"), "pair_hash": pair_hash}
            if cache_token_activations:
                value["token_activations"] = artifacts
            atomic_torch(target, value)
        pooled[question["id"]] = value
        emit("direction_bank_pooled", question=index + 1, total=len(questions), usable=value["usable"])
    return pooled


def validate_token_artifacts(value, output, roles):
    for artifact in value.get("token_activations", {}).values():
        path = output / artifact["path"]
        if not path.exists() or file_sha256(path) != artifact["sha256"]:
            raise ValueError("Committed token activation cache missing or changed")
    if value["usable"] and set(value.get("token_activations", {})) != set(roles):
        raise ValueError("Usable pooled pair is missing required token activations")


def load_token_activation_cache(output, artifact):
    """Load recorded S0 states for new pooling/position analysis, without a model."""
    path = (output / artifact["path"]).resolve()
    if not path.is_relative_to(output.resolve()) or not path.exists() or file_sha256(path) != artifact["sha256"]:
        raise ValueError("Token activation artifact missing or changed")
    record = torch.load(path, weights_only=True, mmap=True)
    length = len(record["student_response_ids"])
    prefix = len(record["student_prompt_ids"])
    if (record.get("schema") != 1 or record["body_mask"].shape != (length,)
            or record["response_offsets"].shape != (length, 2)
            or not torch.equal(record["prediction_state_rows"], torch.arange(length))
            or not torch.equal(record["response_token_state_rows"], torch.arange(1, length + 1))
            or not torch.equal(record["state_input_positions"], torch.arange(prefix - 1, prefix + length))
            or any(value.ndim != 2 or len(value) != length + 1 or not torch.isfinite(value).all()
                   for value in record["states"].values())):
        raise ValueError("Token activation position/tensor contract mismatch")
    return record
