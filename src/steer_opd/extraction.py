"""Frozen-student body-token encoding and explicit HF reference pooling."""
import re
import torch


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


def usable_text(record):
    if record.get("finish_reason") == "length" or not record.get("text", "").strip():
        return False
    words = record["text"].split()
    return len(words) <= 128 or len(set(words[-128:])) >= 8
