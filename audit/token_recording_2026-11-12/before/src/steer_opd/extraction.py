"""Read text pairs in one frozen student checkpoint, without a training graph."""

import re

import torch

from steer_opd.resumable_core import atomic_torch, digest, emit, read_json


def encode_content(tokenizer, text):
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    structural = [m.span() for m in re.finditer(r"</?(?:think|tool_response)>|<\|[^>]+\|>", text)]
    excluded = set(tokenizer.all_special_ids)
    mask = [token not in excluded and b > a and not any(a < end and b > begin for begin, end in structural)
            for token, (a, b) in zip(encoded["input_ids"], encoded["offset_mapping"])]
    return encoded["input_ids"], mask


def pool_question(model, tokenizer, item, records, layers):
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
        for row in records:
            response, mask = encode_content(tokenizer, row["text"])
            if not any(mask):
                return {"usable": False, "reason": "no_answer_content"}
            prefix = item["student_prompt_ids"]
            tokens = torch.tensor([prefix + response], device=device)
            captured.clear()
            with torch.inference_mode():
                model.model(input_ids=tokens, use_cache=False)
            for layer in layers:
                states = captured[layer][0, len(prefix):].float()[torch.tensor(mask, device=device)]
                result[str(layer)]["pools"].append(states.mean(0).cpu())
                result[str(layer)]["norms"].append(states.norm(dim=-1).cpu())
        for record in result.values():
            difference = record["pools"][0] - record["pools"][1]
            if not torch.isfinite(difference).all() or difference.norm() < 1e-8:
                return {"usable": False, "reason": "degenerate_difference_at_candidate_layer"}
        return {"usable": True, "layers": result}
    finally:
        for handle in handles:
            handle.remove()


def extract_pairs(model, tokenizer, questions, role_directories, method, output, identity):
    pooled = {}
    for index, question in enumerate(questions):
        records = [read_json(role_directories[role] / f"{question['id']}.json") for role in method.roles]
        for record in records:
            if record["id"] != question["id"]:
                raise ValueError("Calibration answer belongs to another question")
        pair_hash = digest({"question": question, "records": records, "identity": identity,
                            "layers": method.layers, "roles": method.roles})
        target = output / "pooled" / f"{question['id']}.pt"
        if target.exists():
            value = torch.load(target, weights_only=True)
            if value["pair_hash"] != pair_hash:
                raise ValueError("Calibration texts/checkpoint changed during extraction")
        else:
            value = {**pool_question(model, tokenizer, question, records, method.layers), "pair_hash": pair_hash}
            atomic_torch(target, value)
        pooled[question["id"]] = value
        emit("direction_bank_pooled", question=index + 1, total=len(questions), usable=value["usable"])
    return pooled
