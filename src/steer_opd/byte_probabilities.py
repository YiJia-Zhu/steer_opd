"""Exact native-distribution pushforward to 256 bytes, EOS and control events."""

import hashlib
import json

import torch


BYTE_CLASSES = 258
EOS_CLASS = 256
CONTROL_CLASS = 257
P_METRIC = "P_overlap"


def _device_projection(spec, device):
    cache = spec.setdefault("_device_cache", {})
    key = str(device)
    if key not in cache:
        classes = spec["class_ids"]
        valid = classes >= 0
        cache[key] = {"valid_mask": valid.to(device),
                      "valid_columns": valid.nonzero().flatten().to(device),
                      "classes": classes[valid].to(device),
                      "common_columns": spec["common_ids"].nonzero().flatten().to(device),
                      "all_valid": bool(valid.all())}
    return cache[key]


def projection_spec(codec, other_codec=None):
    """Every defined native ID contributes its entire probability mass once."""
    cache = getattr(codec, "_byte_projection_specs", {})
    cache_key = getattr(other_codec, "mapping_sha256", None)
    if cache_key in cache:
        return cache[cache_key]
    classes = torch.full((codec.support_size,), -1, dtype=torch.long)
    common = torch.zeros(codec.support_size, dtype=torch.bool)
    other_events = ({other_codec.event(t) for t in other_codec.valid_ids} - {None}
                    if other_codec is not None else set())
    for token in codec.valid_ids:
        raw = codec.raw[token]
        classes[token] = (EOS_CLASS if token in codec.eos else CONTROL_CLASS
                          if token in codec.special or not raw else raw[0])
        common[token] = codec.event(token) in other_events if other_codec is not None else False
    identity = {"schema": 1, "tokenizer_mapping_sha256": codec.mapping_sha256,
                "support_size": codec.support_size, "classes": classes.tolist(),
                "common_ids": common.tolist(),
                "class_semantics": "bytes_0_255_EOS256_other_control_or_empty257"}
    sha = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    result = {"schema": 1, "class_ids": classes, "common_ids": common,
            "support_size": codec.support_size, "projection_id": sha,
            "tokenizer_mapping_sha256": codec.mapping_sha256,
            "class_semantics": identity["class_semantics"]}
    cache[cache_key] = result
    codec._byte_projection_specs = cache
    return result


def summarize_logprobs(logps, spec, *, validate=True):
    """GPU scatter-add; no Top-K truncation or transfer of the full vocabulary."""
    if logps.ndim != 2 or logps.shape[-1] != spec["support_size"]:
        raise ValueError("Byte projection must use the declared complete native support")
    view = _device_projection(spec, logps.device)
    probabilities = logps.float().exp()
    # Undefined/padded head IDs must have zero probability, not be assigned a byte.
    if validate and (not torch.isfinite(probabilities).all() or
                     (not view["all_valid"] and (probabilities[:, ~view["valid_mask"]] > 1e-12).any())):
        raise ValueError("Undefined output IDs carry probability in byte projection")
    values = torch.zeros((len(logps), BYTE_CLASSES), device=logps.device, dtype=torch.float32)
    values.scatter_add_(1, view["classes"].expand(len(logps), -1),
                        probabilities if view["all_valid"] else probabilities.index_select(1, view["valid_columns"]))
    if validate:
        validate_byte_probabilities(values)
    entropy = -torch.where(probabilities > 0, probabilities*logps.float(), 0).sum(-1)
    return {"byte_probs": values, "common_vocab_mass": probabilities.index_select(1, view["common_columns"]).sum(-1), "entropy": entropy,
            "projection_id": spec["projection_id"], "byte_projection_schema": 1,
            "byte_class_semantics": spec["class_semantics"]}


def summarize_logits(logits, spec, *, validate=True):
    support = spec["support_size"]
    if logits.ndim != 2 or logits.shape[-1] < support:
        raise ValueError("Native logits are smaller than the declared support")
    values = logits[:, :support].float()
    view = _device_projection(spec, values.device)
    if not view["all_valid"]:
        values = values.masked_fill(~view["valid_mask"], -torch.inf)
    return summarize_logprobs(values.log_softmax(-1), spec, validate=validate)


def validate_byte_probabilities(values, length=None):
    if (values.ndim != 2 or values.shape[-1] != BYTE_CLASSES
            or (length is not None and len(values) != length)
            or not torch.isfinite(values).all() or (values < 0).any()
            or not torch.allclose(values.sum(-1), torch.ones(len(values), device=values.device, dtype=values.dtype), atol=2e-5, rtol=2e-5)):
        raise ValueError("Expected complete, normalized next-byte probabilities including EOS/control")


def validate_projection(record, spec, length):
    if (record.get("projection_id") != spec["projection_id"]
            or record.get("byte_projection_schema") != 1
            or record.get("byte_class_semantics") != spec["class_semantics"]):
        raise ValueError("Byte projection identity differs from the native tokenizer/support")
    validate_byte_probabilities(record["byte_probs"], length)
    mass = record.get("common_vocab_mass")
    entropy = record.get("entropy", record.get("student_entropy"))
    if (mass is None or mass.shape != (length,) or not torch.isfinite(mass).all()
            or ((mass < 0) | (mass > 1+2e-5)).any()
            or entropy is None or entropy.shape != (length,)
            or not torch.isfinite(entropy).all() or (entropy < -1e-6).any()):
        raise ValueError("Full native statistics are incomplete or invalid")


def probability_overlap(student, teacher):
    validate_byte_probabilities(student)
    validate_byte_probabilities(teacher, len(student))
    return torch.minimum(student.double(), teacher.double()).sum(-1).clamp(0, 1)
