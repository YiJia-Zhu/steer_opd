"""V2 runner adapters for full-support statistics with explicit row identities.

Prompt chunks and ordinary sampled rows use upstream InputBatch coordinates.
No sampling decisions, KV state or model forward outputs are changed.
"""

import numpy as np
import torch

from steer_opd.byte_probabilities import summarize_logits, validate_byte_probabilities


def crop_native_support(model, support):
    """Qwen3 has a direct head; Qwen3.5 VL wrappers own a language_model."""
    owner = model
    while not hasattr(owner, "logits_processor") and hasattr(owner, "language_model"):
        owner = owner.language_model
    processor = getattr(owner, "logits_processor", None)
    if processor is None or not 16 <= support <= processor.org_vocab_size:
        raise ValueError("Invalid native output support or unsupported model head layout")
    processor.org_vocab_size = support


class ProjectionCollector:
    def __init__(self, spec):
        self.spec = spec
        self.records = {}

    def add(self, request_id, positions, logits):
        self.add_batch([(request_id, positions, slice(0, len(positions)))], logits)

    def add_batch(self, requests, logits):
        """Reduce a whole engine logit chunk, then transfer only small statistics."""
        if not len(logits):
            return
        stats = summarize_logits(logits, self.spec, validate=False)
        cpu = {k: stats[k].detach().cpu() for k in ("byte_probs", "common_vocab_mass", "entropy")}
        validate_byte_probabilities(cpu["byte_probs"], len(logits))
        for request_id, positions, rows in requests:
            self._add_cpu(request_id, positions, {k: v[rows] for k, v in cpu.items()})

    def _add_cpu(self, request_id, positions, stats):
        if not len(positions):
            return
        byte_probs, common, entropy = (stats[k] for k in ("byte_probs", "common_vocab_mass", "entropy"))
        if len(byte_probs) != len(positions):
            raise ValueError("Engine logit rows differ from request positions")
        record = self.records.setdefault(request_id, {})
        for i, position in enumerate(positions):
            position = int(position)
            value = (byte_probs[i].clone(), common[i].clone(), entropy[i].clone())
            if position in record and any(not torch.allclose(a, b, atol=1e-6, rtol=1e-5)
                                          for a, b in zip(record[position], value)):
                raise ValueError("Repeated engine prediction row has different byte probabilities")
            record[position] = value

    def pop(self, request_id):
        keys = ([request_id] if request_id in self.records else
                [key for key in self.records if key.split("-", 1)[0] == request_id.split("-", 1)[0]])
        if not keys:
            return None
        if len(keys) != 1:
            raise ValueError("Ambiguous engine request identity for projection statistics")
        values = self.records.pop(keys[0])
        positions = sorted(values)
        return {"positions": positions, "byte_probs": torch.stack([values[p][0] for p in positions]),
                "common_vocab_mass": torch.stack([values[p][1] for p in positions]),
                "entropy": torch.stack([values[p][2] for p in positions]),
                "projection_id": self.spec["projection_id"], "byte_projection_schema": 1,
                "byte_class_semantics": self.spec["class_semantics"], "request_id": keys[0]}


def install_projection(runner, spec, *, capture_sampled=True):
    """Install against the pinned V2 runner; unsupported runners fail explicitly."""
    if not hasattr(runner, "prompt_logprobs_worker") or runner.prompt_logprobs_worker is None:
        raise RuntimeError("Full byte statistics require the supported EasySteer V2 runner")
    if capture_sampled and getattr(runner, "batch_sharder", None) is not None:
        raise RuntimeError("Sampled byte collection currently requires unsharded sampling; use one GPU for student source")
    collector = ProjectionCollector(spec)
    original_prompt = runner.prompt_logprobs_worker.compute_prompt_logprobs

    def prompt(logits_fn, hidden_states, batch, all_token_ids, num_computed_tokens, prompt_lens):
        offset = 0

        def logits_with_statistics(states):
            nonlocal offset
            logits = logits_fn(states)
            end = offset + len(states)
            starts = batch.query_start_loc_np
            requests = []
            for i, request_id in enumerate(batch.req_ids):
                a, b = max(offset, int(starts[i])), min(end, int(starts[i+1]))
                if a < b:
                    origin = int(batch.num_computed_tokens_np[i])
                    positions = np.arange(a-int(starts[i])+origin, b-int(starts[i])+origin)
                    requests.append((request_id, positions, slice(a-offset, b-offset)))
            collector.add_batch(requests, logits)
            offset = end
            return logits

        return original_prompt(logits_with_statistics, hidden_states, batch, all_token_ids,
                               num_computed_tokens, prompt_lens)

    runner.prompt_logprobs_worker.compute_prompt_logprobs = prompt
    if capture_sampled:
        original_sample = runner.sample

        def sample(hidden_states, batch, grammar_output):
            if batch.num_draft_tokens:
                raise RuntimeError("Speculative decoding is not accepted by the byte-statistics protocol")
            original_logits = runner.model.compute_logits

            def logits_with_statistics(states):
                logits = original_logits(states)
                indices = batch.logits_indices.detach().cpu().tolist()
                positions = batch.positions.detach().cpu().tolist()
                starts = batch.cu_num_logits_np
                requests = []
                for i, request_id in enumerate(batch.req_ids):
                    a, b = int(starts[i]), int(starts[i+1])
                    requests.append((request_id, [positions[indices[j]] for j in range(a, b)], slice(a, b)))
                collector.add_batch(requests, logits)
                return logits

            runner.model.compute_logits = logits_with_statistics
            try:
                return original_sample(hidden_states, batch, grammar_output)
            finally:
                runner.model.compute_logits = original_logits

        runner.sample = sample
    return collector


def select_prediction_statistics(records, row):
    """One complete native distribution per actual response prediction, t-1."""
    values = [r for r in records if r is not None]
    if not values:
        raise ValueError("The engine did not return full-support byte statistics")
    start = len(row["prompt_token_ids"])-1
    wanted = list(range(start, start+len(row["token_ids"])))
    selected = []
    for record in values:
        mapping = {int(p): i for i, p in enumerate(record["positions"])}
        if any(p not in mapping for p in wanted):
            raise ValueError("Engine byte statistics are missing a source prediction position")
        indices = [mapping[p] for p in wanted]
        result = {k: record[k] for k in ("projection_id", "byte_projection_schema", "byte_class_semantics")}
        result.update(byte_probs=record["byte_probs"][indices], common_vocab_mass=record["common_vocab_mass"][indices],
                      entropy=record["entropy"][indices])
        validate_byte_probabilities(result["byte_probs"], len(wanted))
        selected.append(result)
    first = selected[0]
    for other in selected[1:]:
        if other["projection_id"] != first["projection_id"] or any(
                not torch.allclose(other[k], first[k], atol=1e-6, rtol=1e-5)
                for k in ("byte_probs", "common_vocab_mass", "entropy")):
            raise ValueError("Tensor-parallel byte statistics differ across workers")
    return first
