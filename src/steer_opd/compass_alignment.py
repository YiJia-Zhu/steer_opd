"""Compass Appendix A.4 text-span alignment and explicit overlap adaptation.

Independent implementation from the paper while the requested code URL is
unavailable. Span-mean likelihoods are scores, not complete string-event
probabilities. The byte-continuation Top16 metric below is a project metric,
not an upstream Compass metric or native cross-vocabulary token-ID overlap.
"""

import hashlib
import json

import torch

from steer_opd.resumable_core import digest


METRIC = "rethinking_cross_family_top16_overlap"


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

    def bytes(self, ids):
        try:
            return b"".join(self.raw[int(t)] for t in ids)
        except KeyError as error:
            raise ValueError("Undefined token in source text") from error

    def event(self, token):
        token = int(token)
        # Top16 is a cardinality metric: do not merge distinct EOS IDs or drop
        # matching controls. P_overlap intentionally uses coarser EOS/control classes.
        if token in self.special:
            return "control:"+self.token_names[token]
        if not self.raw.get(token):
            return None
        return "bytes:"+self.raw[token].hex()

    @property
    def event_vocabulary(self):
        if not hasattr(self, "_event_vocabulary"):
            self._event_vocabulary = {self.event(t) for t in self.valid_ids} - {None}
        return self._event_vocabulary


def align_response(source_ids, student_codec, scoring_codecs):
    """Smallest common UTF-8-safe spans; never retokenize the sampled student.

    Special/control tokens and trailing EOS have no scored text span. Their
    alignment mask is False, and their reason is persisted rather than given
    an invented likelihood. Later consistent text spans remain usable.
    """
    source_ids = list(source_ids)
    if not source_ids or "student" in scoring_codecs:
        raise ValueError("Need source tokens and distinct scoring-role names")
    stop = len(source_ids)
    while stop and source_ids[stop-1] in student_codec.eos:
        stop -= 1
    body = source_ids[:stop]
    raw = student_codec.bytes(body)
    text = raw.decode("utf-8", errors="strict")
    codecs = {"student": student_codec, **scoring_codecs}
    ids = {"student": body, **{role: codec.encode(text) for role, codec in scoring_codecs.items()}}
    boundaries = {}
    for role, tokens in ids.items():
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
    safe, offset = {0}, 0
    for char in text:
        offset += len(char.encode("utf-8"))
        safe.add(offset)
    common = sorted(safe.intersection(*(set(v) for v in boundaries.values())))
    mask = [False]*len(source_ids)
    positions = [-1]*len(source_ids)
    units = []
    for start, end in zip(common, common[1:]):
        ranges = {role: [bounds[start], bounds[end]] for role, bounds in boundaries.items()}
        valid = all(not any(t in codecs[role].special for t in ids[role][lo:hi])
                    for role, (lo, hi) in ranges.items())
        unit = {"byte_start": start, "byte_end": end, "ranges": ranges, "valid": valid,
                "reason": None if valid else "control_token_in_span"}
        index = len(units)
        units.append(unit)
        if valid:
            lo, hi = ranges["student"]
            mask[lo:hi] = [True]*(hi-lo)
            positions[lo:hi] = [index]*(hi-lo)
    result = {"schema": 1, "method": "compass_a4_common_byte_boundaries_mean_logp",
              "source_token_ids": source_ids, "native_response_ids": ids,
              "response_utf8_sha256": hashlib.sha256(raw).hexdigest(), "response_text": text,
              "tokenizer_mapping_sha256": {r: c.mapping_sha256 for r, c in codecs.items()},
              "units": units, "alignment_mask": mask, "student_to_unit": positions,
              "terminal_unscored_positions": list(range(stop, len(source_ids)))}
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
    if "teacher" in broadcast and "reference" in broadcast:
        result["teacher_family_shift"] = broadcast["teacher"]-broadcast["reference"]
    return result


def aligned_metrics(plan, student, teacher, student_codec, teacher_codec):
    """Project native Top16 actions to exact next-token bytes at span starts.

    Broadcast to that span's student tokens, matching A.4's credit assignment.
    No renormalization or transported full teacher distribution is implied.
    """
    if plan["tokenizer_mapping_sha256"]["student"] != student_codec.mapping_sha256 or plan["tokenizer_mapping_sha256"]["teacher"] != teacher_codec.mapping_sha256:
        raise ValueError("Metric codec differs from the frozen alignment")
    n = len(plan["source_token_ids"])
    from steer_opd.byte_probabilities import probability_overlap, validate_byte_probabilities
    fields = {k: torch.full((n,), float("nan"), dtype=torch.float64) for k in
              ("overlap", "P_overlap", "student_shared_mass", "teacher_shared_mass",
               "student_event_coverage", "teacher_event_coverage", "student_vocab_coverage",
               "teacher_vocab_coverage", "student_common_vocab_mass", "teacher_common_vocab_mass")}
    def vocabulary(codec):
        if not hasattr(codec, "_metric_event_vocabulary"):
            codec._metric_event_vocabulary = {codec.event(t) for t in codec.valid_ids} - {None}
        return codec._metric_event_vocabulary
    vocabularies = {"student": vocabulary(student_codec), "teacher": vocabulary(teacher_codec)}
    dual = "byte_probs" in student and "byte_probs" in teacher
    if dual:
        validate_byte_probabilities(student["byte_probs"], n)
        validate_byte_probabilities(teacher["byte_probs"], len(plan["native_response_ids"]["teacher"]))
        valid_units = [u for u in plan["units"] if u["valid"]]
        p_values = iter(probability_overlap(
            student["byte_probs"][[u["ranges"]["student"][0] for u in valid_units]],
            teacher["byte_probs"][[u["ranges"]["teacher"][0] for u in valid_units]]).tolist())
    per_unit = []
    def events(codec, score, position):
        result = {}
        for token, lp in zip(score["ids"][position], score["logps"][position]):
            event = codec.event(token)
            if event is not None:
                result[event] = result.get(event, 0.)+float(lp.double().exp())
        return result
    for unit in plan["units"]:
        if not unit["valid"]:
            per_unit.append(None)
            continue
        lo, hi = unit["ranges"]["student"]
        tlo, _ = unit["ranges"]["teacher"]
        a, b = events(student_codec, student, lo), events(teacher_codec, teacher, tlo)
        common = a.keys() & b.keys()
        values = {"overlap": len(common)/16, "student_shared_mass": sum(a[x] for x in common),
                  "teacher_shared_mass": sum(b[x] for x in common),
                  "student_event_coverage": len(a)/16, "teacher_event_coverage": len(b)/16,
                  "student_vocab_coverage": sum(student_codec.event(t) in vocabularies["teacher"] for t in student["ids"][lo])/16,
                  "teacher_vocab_coverage": sum(teacher_codec.event(t) in vocabularies["student"] for t in teacher["ids"][tlo])/16}
        if student_codec.mapping_sha256 == teacher_codec.mapping_sha256:
            # Exact compatible-tokenizer regression, including duplicate byte aliases.
            si, ti = student["ids"][lo], teacher["ids"][tlo]
            values["overlap"] = float((si[:, None] == ti[None, :]).any(-1).double().mean())
        if dual:
            values["P_overlap"] = next(p_values)
            for role, score, pos in [("student", student, lo), ("teacher", teacher, tlo)]:
                if "common_vocab_mass" in score:
                    values[f"{role}_common_vocab_mass"] = float(score["common_vocab_mass"][pos])
        for key, value in values.items():
            fields[key][lo:hi] = value
        per_unit.append(values)
    return {**fields, "metric": METRIC, "metric_version": 2, "P_overlap_available": dual,
            "valid_span_count": sum(bool(u["valid"]) for u in plan["units"]),
            "total_span_count": len(plan["units"]),
            "alignment_id": plan["alignment_id"], "alignment_mask": torch.tensor(plan["alignment_mask"]),
            "unit_metrics": per_unit, "metric_position_semantics": "shared_span_start_broadcast_to_student_tokens",
            "paper_native_overlap_at_16": fields["overlap"] if student_codec.mapping_sha256 == teacher_codec.mapping_sha256 else None,
            "native_token_ids_comparable": student_codec.mapping_sha256 == teacher_codec.mapping_sha256,
            "P_overlap_definition": "full_native_probability_first_byte_pushforward_sum_min",
            "metric_origin": "rethinking_top16_cross_family_adaptation_not_original_native_metric"}
