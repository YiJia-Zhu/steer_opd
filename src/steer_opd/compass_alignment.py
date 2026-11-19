"""Compass Appendix A.4 text-span alignment and explicit overlap adaptation.

Independent implementation from the paper while the requested code URL is
unavailable. Span-mean likelihoods are scores, not complete string-event
probabilities. The prefix-compatible Top16 metric below is a project metric,
not an upstream Compass metric or native cross-vocabulary token-ID overlap.
"""

import hashlib
import json

import torch

from steer_opd.resumable_core import digest


METRIC = "rethinking_cross_family_top16_overlap"
INVALID_UTF8_POLICY = "exact_bytes_invalid_utf8_spans_mask_v1"
VERBATIM_RESPONSE_POLICY = "native_bpe_without_response_normalization_v1"


def _byte_decoder():
    values = list(range(33, 127))+list(range(161, 173))+list(range(174, 256))
    chars = list(values)
    extra = 0
    for value in range(256):
        if value not in values:
            values.append(value)
            chars.append(256+extra)
            extra += 1
    return {chr(char): value for char, value in zip(chars, values)}


class ByteLevelCodec:
    """Recover exact BPE bytes, including tokens that split a UTF-8 codepoint."""

    def __init__(self, tokenizer):
        state = json.loads(tokenizer.backend_tokenizer.to_str())
        if state.get("decoder", {}).get("type") != "ByteLevel":
            raise ValueError("This adapter requires a verified ByteLevel decoder; no heuristic Unicode fallback")
        self.tokenizer = tokenizer
        self.vocab = tokenizer.get_vocab()
        self.token_names = {index: token for token, index in self.vocab.items()}
        self.special = set(tokenizer.all_special_ids)
        self.eos = {tokenizer.eos_token_id} - {None}
        self.eos.update(self.vocab[t] for t in ("<|im_end|>", "<|endoftext|>") if t in self.vocab)
        added = set(tokenizer.get_added_vocab().values())
        decode = _byte_decoder()
        self.raw = {}
        for token, index in self.vocab.items():
            if index in added:
                self.raw[index] = token.encode("utf-8")
            else:
                try:
                    self.raw[index] = bytes(decode[c] for c in token)
                except KeyError as error:
                    raise ValueError(f"Non-ByteLevel base token {token!r}") from error
        self.mapping_sha256 = digest(sorted(self.vocab.items()))
        self.support_size = max(self.raw)+1
        self.valid_ids = sorted(self.raw)
        self.support_id = digest(self.valid_ids)

    def encode(self, text):
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def encode_verbatim(self, text):
        """Encode a response without normalizing its generated Unicode bytes.

        Clone only the backend: prompts and other users of the original
        tokenizer retain its normalizer, special-token handling and settings.
        """
        if not hasattr(self, '_verbatim_backend'):
            from tokenizers import Tokenizer
            state = json.loads(self.tokenizer.backend_tokenizer.to_str())
            state['normalizer'] = None
            self._verbatim_backend = Tokenizer.from_str(json.dumps(state, ensure_ascii=False))
        tokens = list(self._verbatim_backend.encode(text, add_special_tokens=False).ids)
        if self.bytes(tokens) != text.encode('utf-8'):
            raise ValueError('Verbatim response encoding does not preserve exact bytes')
        return tokens

    def bytes(self, ids):
        try:
            return b"".join(self.raw[int(t)] for t in ids)
        except KeyError as error:
            raise ValueError("Undefined token in source text") from error



def _utf8_chunks(raw):
    """Partition bytes into Unicode runs and exact, undecodable byte runs."""
    chunks, chars = [], []
    start = offset = 0
    previous_bad = None
    for char in raw.decode('utf-8', errors='surrogateescape'):
        bad = 0xDC80 <= ord(char) <= 0xDCFF
        if chars and bad != previous_bad:
            chunks.append((start, offset, None if previous_bad else ''.join(chars)))
            chars, start = [], offset
        chars.append(char)
        offset += 1 if bad else len(char.encode('utf-8'))
        previous_bad = bad
    if chars:
        chunks.append((start, offset, None if previous_bad else ''.join(chars)))
    return chunks


def _encode_exact_chunks(codec, raw, chunks, *, verbatim=False):
    """Use native text BPE and native single-byte IDs without replacing bytes."""
    if not hasattr(codec, '_single_byte_token_ids'):
        single = {}
        for token, value in codec.raw.items():
            if len(value) == 1 and token not in codec.special:
                single[value] = min(token, single.get(value, token))
        codec._single_byte_token_ids = single
    result = []
    for start, end, text in chunks:
        if text is not None:
            result.extend(codec.encode_verbatim(text) if verbatim else codec.encode(text))
        else:
            try:
                result.extend(codec._single_byte_token_ids[bytes([byte])] for byte in raw[start:end])
            except KeyError as error:
                raise ValueError('Scorer vocabulary cannot preserve an invalid UTF-8 byte') from error
    return result


def align_response(source_ids, student_codec, scoring_codecs, *, invalid_utf8='raise', response_encoding=None):
    """Smallest common UTF-8-safe spans; never retokenize the sampled student.

    Special/control tokens and trailing EOS have no scored text span. Their
    alignment mask is False, and their reason is persisted rather than given
    an invented likelihood. Later consistent text spans remain usable.

    Static experiments reject malformed UTF-8 by default. Training explicitly
    requests exact-byte scorer inputs and masks only affected common spans.
    A separate verbatim response backend handles tokenizer normalization that
    would otherwise alter generated text; original student IDs are unchanged.
    """
    if invalid_utf8 not in ('raise', 'mask'):
        raise ValueError('invalid_utf8 must be raise or mask')
    # The existing exact-byte training route already opts into UTF-8 masking.
    # Keep its loss caller unchanged so compatible checkpoints can be resumed.
    if response_encoding is None:
        response_encoding = 'verbatim' if invalid_utf8 == 'mask' else 'strict'
    if response_encoding not in ('strict', 'verbatim'):
        raise ValueError('response_encoding must be strict or verbatim')
    source_ids = list(source_ids)
    if not source_ids or "student" in scoring_codecs:
        raise ValueError("Need source tokens and distinct scoring-role names")
    stop = len(source_ids)
    while stop and source_ids[stop-1] in student_codec.eos:
        stop -= 1
    body = source_ids[:stop]
    raw = student_codec.bytes(body)
    bad_ranges = []
    try:
        text = raw.decode("utf-8", errors="strict")
        chunks = [(0, len(raw), text)]
    except UnicodeDecodeError:
        if invalid_utf8 == 'raise':
            raise
        chunks = _utf8_chunks(raw)
        bad_ranges = [[start, end] for start, end, part in chunks if part is None]
        # Display only: scorer IDs below are made from the ORIGINAL bytes.
        # Never use replacement characters or this escaped text as model input.
        text = raw.decode('utf-8', errors='backslashreplace')
    codecs = {"student": student_codec, **scoring_codecs}
    ids = {"student": body, **{role: (_encode_exact_chunks(codec, raw, chunks) if bad_ranges else codec.encode(text))
                              for role, codec in scoring_codecs.items()}}
    boundaries, verbatim_roles = {}, []
    for role, tokens in ids.items():
        if role != 'student' and response_encoding == 'verbatim' and codecs[role].bytes(tokens) != raw:
            tokens = _encode_exact_chunks(codecs[role], raw, chunks, verbatim=True)
            ids[role] = tokens
            verbatim_roles.append(role)
        if codecs[role].bytes(tokens) != raw:
            raise ValueError(f"{role} retokenization does not preserve the exact response bytes")
        ends = {0: 0}
        offset = 0
        for index, token in enumerate(tokens):
            width = len(codecs[role].raw[token])
            if not width:
                raise ValueError("Empty token cannot define a monotone text span")
            offset += width
            ends[offset] = index+1
        boundaries[role] = ends
    safe = {0}
    for start, end, part in chunks:
        safe.update((start, end))
        if part is not None:
            offset = start
            for char in part:
                offset += len(char.encode('utf-8'))
                safe.add(offset)
    common = sorted(safe.intersection(*(set(v) for v in boundaries.values())))
    mask = [False]*len(source_ids)
    positions = [-1]*len(source_ids)
    units = []
    bad_index = 0
    for start, end in zip(common, common[1:]):
        ranges = {role: [bounds[start], bounds[end]] for role, bounds in boundaries.items()}
        while bad_index < len(bad_ranges) and bad_ranges[bad_index][1] <= start:
            bad_index += 1
        bad = bad_index < len(bad_ranges) and bad_ranges[bad_index][0] < end
        valid = not bad and all(not any(t in codecs[role].special for t in ids[role][lo:hi])
                                for role, (lo, hi) in ranges.items())
        unit = {"byte_start": start, "byte_end": end, "ranges": ranges, "valid": valid,
                "reason": None if valid else 'invalid_utf8_in_span' if bad else "control_token_in_span"}
        index = len(units)
        units.append(unit)
        if valid:
            lo, hi = ranges["student"]
            mask[lo:hi] = [True]*(hi-lo)
            positions[lo:hi] = [index]*(hi-lo)
    result = {"schema": 2, "method": "common_utf8_boundaries_mean_native_logp_v2",
              "alignment_roles": list(codecs),
              "source_token_ids": source_ids, "native_response_ids": ids,
              "response_utf8_sha256": hashlib.sha256(raw).hexdigest(), "response_text": text,
              "tokenizer_mapping_sha256": {r: c.mapping_sha256 for r, c in codecs.items()},
              "units": units, "alignment_mask": mask, "student_to_unit": positions,
              "terminal_unscored_positions": list(range(stop, len(source_ids)))}
    if bad_ranges:
        result['method'] = 'common_utf8_boundaries_exact_bytes_invalid_mask_v3'
        result['invalid_utf8'] = {'policy': INVALID_UTF8_POLICY, 'byte_ranges': bad_ranges,
                                  'bytes': sum(end-start for start, end in bad_ranges),
                                  'response_text_format': 'backslash_escaped_display_only'}
    if verbatim_roles:
        result['verbatim_response_encoding'] = {'policy': VERBATIM_RESPONSE_POLICY, 'roles': verbatim_roles}
    result["alignment_id"] = digest(result)
    return result


def _validate_scores(plan, records):
    for role, ids in plan["native_response_ids"].items():
        if role not in records:
            raise ValueError(f"Missing {role} span scores")
        length = len(plan["source_token_ids"]) if role == "student" else len(ids)
        record = records[role]
        if (record["sampled_logps"].shape != (length,) or record["ids"].shape != (length, 16)
                or record["logps"].shape != (length, 16)
                or not torch.isfinite(record["sampled_logps"]).all()):
            raise ValueError(f"{role} probability positions differ from the alignment plan")


def span_scores(plan, records):
    """Average each model's native logps within the same observed text unit."""
    _validate_scores(plan, records)
    n = len(plan["source_token_ids"])
    broadcast = {role: torch.full((n,), float("nan")) for role in records}
    units = []
    for unit in plan["units"]:
        scores = {}
        if unit["valid"]:
            lo, hi = unit["ranges"]["student"]
            for role, (start, end) in unit["ranges"].items():
                value = records[role]["sampled_logps"][start:end].float().mean()
                scores[role] = float(value)
                broadcast[role][lo:hi] = value
        units.append(scores)
    result = {"unit_mean_logps": units, "broadcast_mean_logps": broadcast,
              "alignment_mask": torch.tensor(plan["alignment_mask"]),
              "aggregation": "mean_native_token_logp_within_shared_span_not_string_probability"}
    return result


def aligned_metrics(plan, student, teacher, student_codec, teacher_codec):
    """Maximum compatible candidate/mass matching at common span starts."""
    if plan["tokenizer_mapping_sha256"]["student"] != student_codec.mapping_sha256 or plan["tokenizer_mapping_sha256"]["teacher"] != teacher_codec.mapping_sha256:
        raise ValueError("Metric codec differs from the frozen alignment")
    from steer_opd.prefix_matching import (maximum_topk_matching, probability_overlap,
        METRIC_VERSION, RULE_VERSION, P_DEFINITION, TOPK_TIE_POLICY, TOP_K)
    from steer_opd.static_masks import validate_native_score
    n = len(plan["source_token_ids"])
    validate_native_score(student, n, student_codec.support_size)
    validate_native_score(teacher, len(plan["native_response_ids"]["teacher"]), teacher_codec.support_size)
    fields = {k: torch.full((n,), float("nan"), dtype=torch.float64) for k in
              ("overlap", "P_overlap", "student_shared_mass", "teacher_shared_mass",
               "student_top16_mass", "teacher_top16_mass")}
    valid_units = [u for u in plan["units"] if u["valid"]]
    p_values = iter(probability_overlap(student, teacher, student_codec, teacher_codec,
        student_positions=[u["ranges"]["student"][0] for u in valid_units],
        teacher_positions=[u["ranges"]["teacher"][0] for u in valid_units]).tolist())
    per_unit = []
    for unit in plan["units"]:
        if not unit["valid"]:
            per_unit.append(None)
            continue
        lo, hi = unit["ranges"]["student"]
        tlo, _ = unit["ranges"]["teacher"]
        pairs = maximum_topk_matching(student["ids"][lo], teacher["ids"][tlo], student_codec, teacher_codec)
        sp = student["logps"][lo].double().exp()
        tp = teacher["logps"][tlo].double().exp()
        values = {"overlap": len(pairs)/TOP_K, "P_overlap": next(p_values),
                  "student_shared_mass": sum(float(sp[i]) for i, _ in pairs),
                  "teacher_shared_mass": sum(float(tp[j]) for _, j in pairs),
                  "student_top16_mass": float(sp.sum()), "teacher_top16_mass": float(tp.sum()),
                  "top16_matching_indices": pairs}
        for key, value in values.items():
            if key in fields:
                fields[key][lo:hi] = value
        per_unit.append(values)
    return {**fields, "metric": METRIC, "metric_version": METRIC_VERSION, "P_overlap_available": True,
            "matching_rule_version": RULE_VERSION, "topk_tie_policy": TOPK_TIE_POLICY, "returned_top_k": TOP_K,
            "shared_mass_definition": "mass_of_deterministic_maximum_cardinality_matching_not_P_overlap",
            "valid_span_count": sum(bool(u["valid"]) for u in plan["units"]),
            "total_span_count": len(plan["units"]),
            "alignment_id": plan["alignment_id"], "alignment_mask": torch.tensor(plan["alignment_mask"]),
            "unit_metrics": per_unit, "metric_position_semantics": "shared_span_start_broadcast_to_student_tokens",
            "paper_native_overlap_at_16": fields["overlap"] if student_codec.mapping_sha256 == teacher_codec.mapping_sha256 else None,
            "native_token_ids_comparable": student_codec.mapping_sha256 == teacher_codec.mapping_sha256,
            "P_overlap_definition": P_DEFINITION,
            "metric_origin": "rethinking_top16_cross_family_adaptation_not_original_native_metric"}
