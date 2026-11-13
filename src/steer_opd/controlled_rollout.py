"""Cached HF reference backend for causal gates and exact intervention replay.

The fast all-position branch remains vLLM. This backend intentionally rejects
cache types whose updates may overwrite historical states in place.
"""

import copy
from contextlib import nullcontext

import torch
from transformers.cache_utils import DynamicLayer

from steer_opd.resumable_core import project_logits
from steer_opd.steering import response_prediction_mask


HF_SOURCE = "hf_controlled_rollout_raw_logprobs"


def fork_cache(cache):
    if cache is None:
        return None
    if not hasattr(cache, "layers") or any(type(layer) is not DynamicLayer for layer in cache.layers):
        raise ValueError("Probe isolation currently requires append-only DynamicLayer caches")
    branch = copy.copy(cache)
    branch.layers = [copy.copy(layer) for layer in cache.layers]
    # DynamicLayer.update concatenates new tensors, never mutating old keys or
    # values. Copying the layer containers isolates the probe without copying KV.
    return branch


def next_distribution(model, token_ids, cache, controller=None, inject=False):
    device = model.lm_head.weight.device
    tokens = torch.tensor([token_ids], device=device)
    mask = torch.zeros_like(tokens, dtype=torch.bool)
    mask[:, -1] = bool(inject)
    scope = controller.loss_scope(mask) if controller and inject else nullcontext()
    with scope:
        output = model.model(input_ids=tokens, past_key_values=cache, use_cache=True)
    return project_logits(model, output.last_hidden_state[:, -1])[0].log_softmax(-1), output.past_key_values


def gate_decision(method, position, probe=None, teacher=None, *, random_draw=None, gate_history=None):
    if method.gate == "static_overlap":
        raise ValueError("Static overlap needs a frozen source mask; use the static scoring/engine entrypoint")
    metrics = {}
    if method.gate == "all":
        return True, metrics
    if method.gate == "late":
        return position >= method.start_token, metrics
    if method.gate == "random":
        if random_draw is None:
            raise ValueError("Random gate requires a dedicated causal RNG draw")
        return random_draw < method.random_probability, metrics
    if probe is None or teacher is None or probe.shape != teacher.shape:
        raise ValueError("Online gate requires same-support probe and teacher distributions")
    if method.gate in ("overlap", "matched_random", "matched_position_random"):
        k = min(16, len(probe))
        overlap = float((probe.topk(k).indices[:, None] == teacher.topk(k).indices[None, :]).any(-1).float().mean())
        metrics["probe_overlap16"] = overlap
        if method.gate != "overlap":
            from steer_opd.gate_controls import matched_draw
            if gate_history is None or random_draw is None:
                raise ValueError("Matched gate needs resumable causal history and RNG")
            inject, matching = matched_draw(gate_history, overlap < method.overlap_threshold, position,
                random_draw, method.random_probability, method.matching_bin_width,
                method.gate == "matched_position_random")
            return inject, {**metrics, **matching}
        return overlap < method.overlap_threshold, metrics
    sh = float(-(probe.exp() * probe).sum())
    th = float(-(teacher.exp() * teacher).sum())
    metrics.update(probe_entropy=sh, teacher_entropy=th, signed_entropy_gap=sh - th)
    return sh - th > method.entropy_threshold, metrics


def controlled_step(student, teacher, tokens, student_cache, teacher_cache, controller, method, position, draw, gate_history=None,
                    *, record_probe=False):
    q, teacher_cache = next_distribution(teacher, tokens, teacher_cache)
    probe, probe_cache = None, None
    if method.online_gate or record_probe:
        # Both branches start from the identical controlled history. The probe's
        # upper-layer KV never enters the official branch after a triggered gate.
        probe, probe_cache = next_distribution(student, tokens, fork_cache(student_cache))
        if not method.online_gate:
            del probe_cache
            probe_cache = None
    inject, metrics = gate_decision(method, position, probe, q, random_draw=draw, gate_history=gate_history)
    if method.online_gate and not inject:
        p, student_cache = probe, probe_cache
    else:
        p, student_cache = next_distribution(student, tokens, student_cache, controller, inject)
    return p, q, student_cache, teacher_cache, inject, metrics, probe


def rebuild_history(model, row, controller=None):
    """Rebuild KV once after interruption, using the saved historical mask."""
    if not row["token_ids"]:
        return None
    prefix, response = row["prompt_token_ids"], row["token_ids"]
    device = model.lm_head.weight.device
    tokens = torch.tensor([prefix + response[:-1]], device=device)
    mask = response_prediction_mask(len(prefix), len(response), row.get("steering_mask"), device=device)
    scope = controller.loss_scope(mask) if controller else nullcontext()
    with scope:
        return model.model(input_ids=tokens, use_cache=True).past_key_values


def generate_response(student, teacher, request, controller, method, *, max_tokens, support_size, k,
                      stop_ids, progress=None, save_progress=None, save_every=32, audit_full=False,
                      record_probe=False):
    """One request, resumable at token boundaries. No parameter update or graph."""
    device = student.lm_head.weight.device
    sampling = torch.Generator(device=device).manual_seed(request["seed"])
    gating = torch.Generator().manual_seed(request["seed"] + 87654321)
    state = progress or {"token_ids": [], "steering_mask": [], "gate_metrics": [],
                         "ids": [], "logps": [], "sampled_logps": [],
                         "teacher_logps": [], "teacher_topk_ids": [], "teacher_topk_logps": [],
                         "probe_topk_ids": [], "probe_topk_logps": [],
                         "student_entropy": [], "teacher_entropy": [], "probe_entropy": []}
    state.setdefault("gate_history", {})
    if progress and "student_entropy" not in state:
        raise ValueError("Partial rollout predates required token statistics; do not silently invent missing entropy")
    if audit_full:
        state.setdefault("full_rollout_logps", [])
        state.setdefault("full_teacher_logps", [])
        state.setdefault("full_probe_logps", [])
    with torch.inference_mode():
        if progress:
            sampling.set_state(state["sampling_rng"])
            gating.set_state(state["gating_rng"])
            row = {**request, **state}
            student_cache = rebuild_history(student, row, controller)
            teacher_cache = rebuild_history(teacher, row)
        else:
            student_cache, teacher_cache = None, None
        while len(state["token_ids"]) < max_tokens:
            if state["token_ids"] and state["token_ids"][-1] in stop_ids:
                break
            tokens = [state["token_ids"][-1]] if state["token_ids"] else request["prompt_token_ids"]
            p, q, student_cache, teacher_cache, inject, metrics, probe = controlled_step(
                student, teacher, tokens, student_cache, teacher_cache, controller, method,
                len(state["token_ids"]), float(torch.rand((), generator=gating)), state["gate_history"],
                record_probe=record_probe)
            if controller is None:
                inject = False
            if len(p) != support_size or len(q) != support_size:
                raise ValueError("Teacher/student must use exactly the frozen output support")
            if not torch.isfinite(p).all() or not torch.isfinite(q).all():
                raise FloatingPointError("Nonfinite rollout distribution")
            probabilities = p.exp()
            token = int(torch.multinomial(probabilities, 1, generator=sampling))
            values, ids = p.topk(k)
            tv, ti = q.topk(k)
            state["token_ids"].append(token)
            state["steering_mask"].append(bool(inject))
            state["gate_metrics"].append(metrics)
            state["ids"].append(ids.cpu().int())
            state["logps"].append(values.cpu().float())
            state["sampled_logps"].append(float(p[token]))
            state["student_entropy"].append(float(-(probabilities * p).sum()))
            state["teacher_entropy"].append(float(-(q.exp() * q).sum()))
            if audit_full:
                state["full_rollout_logps"].append(p.cpu())
                state["full_teacher_logps"].append(q.cpu())
                if probe is not None:
                    state["full_probe_logps"].append(probe.cpu())
            state["teacher_logps"].append(q[ids].cpu().float())
            state["teacher_topk_ids"].append(ti.cpu().int())
            state["teacher_topk_logps"].append(tv.cpu().float())
            if probe is not None:
                pv, pi = probe.topk(k)
                state["probe_topk_ids"].append(pi.cpu().int())
                state["probe_topk_logps"].append(pv.cpu().float())
                state["probe_entropy"].append(float(-(probe.exp() * probe).sum()))
            if save_progress and len(state["token_ids"]) % save_every == 0:
                state["sampling_rng"], state["gating_rng"] = sampling.get_state(), gating.get_state()
                save_progress(state)
    state["sampling_rng"], state["gating_rng"] = sampling.get_state(), gating.get_state()
    if not state["token_ids"]:
        raise ValueError("Empty controlled rollout")
    score = {key: torch.stack(state[key]) for key in ["ids", "logps", "teacher_logps", "teacher_topk_ids", "teacher_topk_logps"]}
    for key in ["probe_topk_ids", "probe_topk_logps"]:
        if state[key]:
            score[key] = torch.stack(state[key])
    if audit_full:
        score["full_rollout_logps"] = torch.stack(state["full_rollout_logps"])
        score["full_teacher_logps"] = torch.stack(state["full_teacher_logps"])
        if state["full_probe_logps"]:
            score["full_probe_logps"] = torch.stack(state["full_probe_logps"])
    for key in ["student_entropy", "teacher_entropy", "probe_entropy"]:
        if state[key]:
            score[key] = torch.tensor(state[key], dtype=torch.float32)
    score["entropy_support_size"] = support_size
    score.update(source=HF_SOURCE, k=k, support_size=support_size, probability_mode="raw_logprobs",
                 sampled_logps=torch.tensor(state["sampled_logps"]))
    row = {**request, "token_ids": state["token_ids"], "steering_mask": state["steering_mask"],
           "gate_metrics": state["gate_metrics"], "sampled_logps": state["sampled_logps"],
           "finish_reason": "stop" if state["token_ids"][-1] in stop_ids else "length",
           "token_record_schema": 1, "prediction_start": len(request["prompt_token_ids"]) - 1}
    return row, score
