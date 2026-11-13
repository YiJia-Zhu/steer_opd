"""Batched exact causal probes, with per-request vectors and isolated KV.

This is a PyTorch backend, not an EasySteer implementation. It is used for a
common-backend gate panel until an engine-native two-pass gate is validated.
"""

import copy
from contextlib import nullcontext

import torch

from steer_opd.controlled_rollout import fork_cache, gate_decision
from steer_opd.resumable_core import project_logits


class BatchMemoryPressure(RuntimeError):
    pass


def available_memory():
    if not torch.cuda.is_available():
        return float("inf")
    free, _ = torch.cuda.mem_get_info()
    return free + torch.cuda.memory_reserved() - torch.cuda.memory_allocated()


def cache_bytes(cache):
    if cache is None:
        return 0
    return sum(value.numel()*value.element_size() for layer in cache.layers if layer.is_initialized
               for value in (layer.keys, layer.values))


def require_memory(additional):
    if available_memory() < additional + 2*1024**3:
        raise BatchMemoryPressure("Split pending requests before temporary KV tensors exhaust memory")


class BatchedController:
    def __init__(self, model, vectors, *, operator="additive", alpha=1.0, record_perturbations=False):
        from steer_opd.operators import validate_operator
        validate_operator(operator, alpha)
        if not vectors or any(value.ndim != 2 or not torch.isfinite(value).all() for value in vectors.values()):
            raise ValueError("Batched vectors must be finite [batch,hidden] tensors")
        self.mask = None
        self.indices = None
        self.handles = []
        self.perturbations = {}
        self.nonzero_rows = torch.stack([value.float().norm(dim=-1)>0 for value in vectors.values()]).any(0).tolist()
        for layer, delta in vectors.items():
            def hook(module, args, output, delta=delta, layer_index=layer):
                if self.mask is None:
                    return output
                hidden = output[0] if isinstance(output, tuple) else output
                values = delta if self.indices is None else delta[self.indices]
                if hidden.shape[:2] != self.mask.shape or hidden.shape[0] != values.shape[0]:
                    raise ValueError("Batched steering shape mismatch")
                if operator == "additive":
                    delta = values.to(hidden)[:, None, :]
                    if alpha != 1:
                        delta = delta * alpha
                    changed = hidden + self.mask.to(hidden.dtype)[..., None] * delta
                else:
                    from steer_opd.operators import slerp
                    changed = torch.where(self.mask[..., None], slerp(hidden, values[:, None, :], alpha), hidden)
                if record_perturbations:
                    self.perturbations[layer_index] = ((changed.float()-hidden.float()).norm(dim=-1) /
                        hidden.float().norm(dim=-1).clamp_min(1e-8)).detach()
                return (changed, *output[1:]) if isinstance(output, tuple) else changed
            self.handles.append(model.model.layers[layer].register_forward_hook(hook))

    def scope(self, mask, indices=None):
        from contextlib import contextmanager
        @contextmanager
        def active():
            if self.mask is not None:
                raise ValueError("Nested batched intervention")
            self.mask, self.indices = mask, indices
            try:
                yield
            finally:
                self.mask, self.indices = None, None
        return active()

    def detach(self):
        for handle in self.handles:
            handle.remove()


def select_cache(cache, indices):
    if cache is None:
        return None
    result = fork_cache(cache)
    for layer in result.layers:
        if layer.is_initialized:
            layer.keys = layer.keys.index_select(0, indices)
            layer.values = layer.values.index_select(0, indices)
    return result


def forward_rows(model, tokens, cache, attention_mask, controller=None, injection=None, indices=None):
    positions = attention_mask.long().cumsum(-1).sub(1).clamp_min(0)[:, -tokens.shape[1]:]
    scope = controller.scope(injection, indices) if controller is not None and injection is not None else nullcontext()
    with scope:
        result = model.model(input_ids=tokens, attention_mask=attention_mask, position_ids=positions,
                             past_key_values=cache, use_cache=True)
    return project_logits(model, result.last_hidden_state[:, -1]).log_softmax(-1), result.past_key_values


def batched_step(student, teacher, tokens, student_cache, teacher_cache, attention_mask,
                 controller, method, positions, draws, histories, active, *, record_probe=False):
    if tokens.shape[1] != 1:
        raise ValueError("Batched gate step requires one pending token per request; prefill history separately")
    q, teacher_cache = forward_rows(teacher, tokens, teacher_cache, attention_mask)
    probe = probe_cache = None
    if method.online_gate or record_probe:
        require_memory(cache_bytes(student_cache))
        probe, probe_cache = forward_rows(student, tokens, fork_cache(student_cache), attention_mask)
        if not method.online_gate:
            # Diagnostic only: the original formal branch still starts from
            # its unchanged cache and runs at its original batch shape.
            del probe_cache
            probe_cache = None
    decisions, metrics = [], []
    for index, position in enumerate(positions):
        if not active[index]:
            decisions.append(False)
            metrics.append({})
            continue
        decision, metric = gate_decision(method, position,
            probe[index] if probe is not None else None, q[index], random_draw=draws[index],
            gate_history=histories[index])
        effective = decision and controller is not None and controller.nonzero_rows[index]
        if decision and controller is not None and not controller.nonzero_rows[index]:
            metric = {**metric, "gate_requested": True, "zero_vector_no_intervention": True}
        decisions.append(bool(effective))
        metrics.append(metric)
    mask = torch.tensor(decisions, device=tokens.device, dtype=torch.bool)
    if not method.online_gate:
        injection = torch.zeros_like(tokens, dtype=torch.bool)
        injection[:, -1] = mask
        p, student_cache = forward_rows(student, tokens, student_cache, attention_mask, controller, injection)
    else:
        p, student_cache = probe, probe_cache
        selected = mask.nonzero().flatten()
        if selected.numel():
            require_memory(cache_bytes(student_cache) * len(selected)/len(tokens))
            # Probe cache is a new append-only tensor. Replace only its current
            # slot for triggered rows; historical KV is unchanged in both paths.
            old = select_cache(student_cache_before := student_cache, selected)
            # Remove the probe's current slot to recover the original history.
            for layer in old.layers:
                layer.keys = layer.keys[..., :-1, :]
                layer.values = layer.values[..., :-1, :]
            injection = torch.ones((len(selected), 1), dtype=torch.bool, device=tokens.device)
            formal, formal_cache = forward_rows(student, tokens[selected], old, attention_mask[selected],
                                                controller, injection, selected)
            p = p.clone()
            p[selected] = formal
            for merged, corrected in zip(student_cache_before.layers, formal_cache.layers):
                merged.keys[selected, :, -1:, :] = corrected.keys[..., -1:, :]
                merged.values[selected, :, -1:, :] = corrected.values[..., -1:, :]
    return p, q, student_cache, teacher_cache, decisions, metrics, probe
