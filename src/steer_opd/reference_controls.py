"""One common text pool/filter for all three direction decompositions."""

import re

import torch

from steer_opd.extraction import encode_content
from steer_opd.resumable_core import digest


class BootstrapQualityError(ValueError):
    """A bounded native-student text budget did not yield usable references."""


def usable_text(record):
    if record.get("finish_reason") == "length" or not record.get("text", "").strip():
        return False
    words = record["text"].split()
    return len(words) <= 128 or len(set(words[-128:])) >= 8


def common_records(records, tokenizer, *, normalize_template=False, match_length=False, correctness=None):
    if set(records) != {"teacher", "reference", "student"}:
        raise ValueError("Three-direction comparison requires the same T/R/S text triplet")
    if not all(usable_text(value) for value in records.values()):
        return None, "common_text_filter"
    if correctness is not None and not all(correctness.get(role, False) for role in ("teacher", "reference")):
        return None, "teacher_reference_not_both_correct"
    result = {role: dict(record) for role, record in records.items()}
    for role, record in result.items():
        record["source_record_hash"] = digest(records[role])
        if normalize_template:
            text = re.sub(r"</?(?:think|tool_response)>|<\|[^>]+\|>", "", record["text"])
            record["text"] = text.strip()
    if match_length:
        encoded = {role: encode_content(tokenizer, row["text"])[0] for role, row in result.items()}
        length = min(map(len, encoded.values()))
        if length < 1:
            return None, "empty_common_length"
        for role, row in result.items():
            row["text"] = tokenizer.decode(encoded[role][:length], skip_special_tokens=False)
            row["length_control"] = "same_student_token_prefix_length"
    tokenizations = [tuple(encode_content(tokenizer, row["text"])[0]) for row in result.values()]
    if len(set(tokenizations)) < 3:
        return None, "identical_reference_tokenizations_degenerate_direction"
    return result, None


def direction_geometry(pools):
    """pools maps roles to tensors from the SAME extractor, question and layer."""
    t, r, s = (pools[role].float() for role in ("teacher", "reference", "student"))
    directions = {"T-S": t-s, "T-R": t-r, "R-S": r-s}
    eps = 1e-12
    return {"raw_norms": {key: float(value.norm()) for key, value in directions.items()},
            "cosines": {f"{a}:{b}": float(torch.dot(directions[a], directions[b]) /
                        (directions[a].norm() * directions[b].norm()).clamp_min(eps))
                        for a, b in [("T-S", "T-R"), ("T-S", "R-S"), ("T-R", "R-S")]},
            "decomposition_max_error": float((directions["T-S"]-directions["T-R"]-directions["R-S"]).abs().max()),
            "cancellation_ratio": float(directions["T-S"].norm() /
                (directions["T-R"].norm()+directions["R-S"].norm()).clamp_min(eps))}
