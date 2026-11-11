"""Diagnostics on shared decoded continuations and first UTF-8 bytes.

Literal next-token continuation overlap depends on tokenization. First-byte
projection is deliberately coarse. Neither metric alone establishes reasoning
compatibility or supplies a cross-tokenizer OPD loss.
"""

import math


def visible_continuation(tokenizer, prompt_ids, response_ids):
    """Decode in context: SentencePiece can drop an initial space in isolation."""
    kwargs = {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}
    prompt_text = tokenizer.decode(prompt_ids, **kwargs)
    full_text = tokenizer.decode(prompt_ids + response_ids, **kwargs)
    if not full_text.startswith(prompt_text):
        raise ValueError("Response decoding changed the visible prompt boundary")
    return full_text[len(prompt_text):]


def _projection(record):
    mass = {}
    for token in record["top_tokens"]:
        key = token.get("byte_event")
        if key is not None:
            mass[key] = mass.get(key, 0.0) + token["probability"]
    residual = max(0.0, 1.0 - sum(mass.values()))
    return mass, residual


def compare_distributions(teacher, student, k=16):
    a, ra = _projection(teacher)
    b, rb = _projection(student)
    support = set(a) | set(b)
    lower = min(1.0, sum(min(a.get(x, 0.0), b.get(x, 0.0)) for x in support))
    # Unobserved mass can match either model's already-observed mass. Using
    # min(ra, rb) here would give an invalid upper bound.
    upper = min(1.0, lower + ra + rb)
    with_residual_a, with_residual_b = dict(a), dict(b)
    with_residual_a["residual"] = ra
    with_residual_b["residual"] = rb
    jsd = 0.0
    for event in support | {"residual"}:
        p, q = with_residual_a.get(event, 0.0), with_residual_b.get(event, 0.0)
        midpoint = (p + q) / 2
        if p:
            jsd += 0.5 * p * math.log(p / midpoint)
        if q:
            jsd += 0.5 * q * math.log(q / midpoint)
    literal_a = {x["continuation_hex"] for x in teacher["top_tokens"][:k]
                 if x.get("continuation_hex") is not None}
    literal_b = {x["continuation_hex"] for x in student["top_tokens"][:k]
                 if x.get("continuation_hex") is not None}
    same_vocab = teacher["tokenizer_mapping_sha256"] == student["tokenizer_mapping_sha256"]
    native = None
    if same_vocab:
        native = len({x["token_id"] for x in teacher["top_tokens"][:k]} &
                     {x["token_id"] for x in student["top_tokens"][:k]}) / k
    return {
        "paper_native_overlap_at_16": native,
        "native_token_ids_comparable": same_vocab,
        "literal_continuation_overlap_at_16": len(literal_a & literal_b) / k,
        "teacher_valid_literal_topk": len(literal_a),
        "student_valid_literal_topk": len(literal_b),
        "first_byte_overlap_lower_bound": lower,
        "first_byte_overlap_upper_bound": upper,
        "first_byte_jsd_with_residual_bucket_nats": jsd,
        "teacher_unresolved_probability_mass": ra,
        "student_unresolved_probability_mass": rb,
    }
