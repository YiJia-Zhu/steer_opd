"""Single-GPU primitives for the audited Student Top-K OPD experiment."""

from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from steer_opd.steering import (SteeringController, CompositeSteeringController, find_decoder_layer,
                                response_prediction_mask)


ROOT = Path(__file__).resolve().parents[2]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def emit(event, **fields):
    print(json.dumps({"event": event, "time": time.time(), **fields}, ensure_ascii=False), flush=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def require_gpu():
    from steer_opd.runtime import verify_runtime
    verify_runtime()
    if not os.environ.get("STEER_OPD_GPU_ADMISSION"):
        raise RuntimeError("Fresh idle-GPU admission is required")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("This backend must see exactly one admitted GPU")


def load_model(path, train=False):
    from transformers import AutoModelForCausalLM

    require_gpu()
    model = AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation=os.environ.get("STEER_OPD_HF_ATTENTION", "sdpa"), device_map="cuda:0",
    )
    model.config.use_cache = not train
    model._opd_vocab_size = int(os.environ.get("STEER_OPD_SUPPORT_SIZE", model.config.vocab_size))
    model.requires_grad_(train)
    if train:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
    else:
        model.eval()
    return model


def controller_for(model, vector_path, row=None):
    if not vector_path:
        return None
    payload = torch.load(vector_path, map_location="cpu", weights_only=True)
    if payload.get("schema") == 2:
        from steer_opd.direction_bank import bank_vectors, file_sha256
        if row and row.get("vector_hash") and file_sha256(vector_path) != row["vector_hash"]:
            raise ValueError("Direction bank changed after rollout")
        vectors, vector_id = bank_vectors(payload, row.get("id") if row else None)
        if row and row.get("vector_id") != vector_id:
            raise ValueError("Rollout vector identity mismatch")
        operator = payload["method"].get("operator", "additive")
        return CompositeSteeringController(model, vectors, operator=operator,
            alpha=payload["method"]["alpha"] if operator == "slerp" else 1.0)
    return SteeringController(find_decoder_layer(model, payload["layer"]), payload["delta"], 1.0)


def response_hidden(model, row, controller=None):
    """Return states predicting every actual response token, including EOS."""
    prefix, response = row["prompt_token_ids"], row["token_ids"]
    if not prefix or not response:
        raise ValueError("Empty prompt/response")
    device = model.lm_head.weight.device
    tokens = torch.tensor([prefix + response[:-1]], dtype=torch.long, device=device)
    mask = response_prediction_mask(len(prefix), len(response), row.get("steering_mask"), device=device)
    scope = controller.loss_scope(mask) if controller else nullcontext()
    # Caller performing backward must open its own persistent scope. This helper
    # is intentionally inference-only; training uses response_forward below.
    with scope:
        result = model.model(input_ids=tokens, use_cache=False).last_hidden_state
    return result[0, len(prefix) - 1:]


class FP32OutputLinear(torch.autograd.Function):
    """FP32-output BF16 GEMM with an explicit, tested FP32 backward.

    The installed torch implements mm(out_dtype=...) for inference but has no
    autograd rule for that overload. The derivatives are ordinary linear ones.
    """

    @staticmethod
    def forward(ctx, hidden, weight):
        ctx.save_for_backward(hidden, weight)
        if hidden.is_cuda and hidden.dtype == torch.bfloat16:
            return torch.mm(hidden, weight.t(), out_dtype=torch.float32)
        dtype = torch.float32 if hidden.dtype == torch.bfloat16 else hidden.dtype
        return torch.nn.functional.linear(hidden.to(dtype), weight.to(dtype))

    @staticmethod
    def backward(ctx, grad):
        hidden, weight = ctx.saved_tensors
        grad_hidden = torch.mm(grad, weight.to(grad.dtype)).to(hidden.dtype) if ctx.needs_input_grad[0] else None
        grad_weight = torch.mm(grad.t(), hidden.to(grad.dtype)).to(weight.dtype) if ctx.needs_input_grad[1] else None
        return grad_hidden, grad_weight


def project_logits(model, hidden, full_vocab=False):
    # Match vLLM head_dtype=float32: BF16 operands, FP32 accumulator/output.
    # Avoid BF16 rounding of large logits before log-softmax (RL consistency).
    flat = hidden.reshape(-1, hidden.shape[-1])
    weight = model.lm_head.weight
    if flat.device != weight.device:
        flat = flat.to(weight.device)
    if not full_vocab:
        weight = weight[:getattr(model, "_opd_vocab_size", weight.shape[0])]
    if torch.is_grad_enabled() and (flat.requires_grad or weight.requires_grad):
        logits = FP32OutputLinear.apply(flat, weight)
    elif flat.is_cuda and flat.dtype == torch.bfloat16:
        logits = torch.mm(flat, weight.t(), out_dtype=torch.float32)
    else:
        logits = torch.nn.functional.linear(flat.float(), weight.float())
    return logits.reshape(*hidden.shape[:-1], -1)


def score_hidden(model, hidden, k=16, candidates=None, chunk=128, sampled=None, own_topk=False, include_entropy=False):
    ids, logps, sampled_lps, undefined_mass, excluded_mass = [], [], [], [], []
    own_ids, own_logps = [], []
    entropies = []
    token_vocab_size = getattr(model, "_defined_vocab_size", model.config.vocab_size)
    for start in range(0, len(hidden), chunk):
        logits = project_logits(model, hidden[start:start + chunk], full_vocab=True)
        support = getattr(model, "_opd_vocab_size", logits.shape[-1])
        raw_lp = logits.log_softmax(-1)
        supported = logits[:, :support]
        valid_ids = getattr(model, "_opd_valid_output_ids", None)
        if valid_ids is not None:
            valid = getattr(model, "_opd_valid_output_mask", None)
            if valid is None or valid.shape != (support,) or valid.device != logits.device:
                valid = torch.zeros(support, dtype=torch.bool, device=logits.device)
                valid[torch.as_tensor(valid_ids, device=logits.device)] = True
                model._opd_valid_output_mask = valid
            supported = supported.masked_fill(~valid, -1e30)
        lp = supported.log_softmax(-1)
        if include_entropy:
            entropies.append((-(lp.exp() * lp).sum(-1)).cpu())
        if candidates is None:
            values, selected = lp.topk(k, dim=-1)
        else:
            selected = candidates[start:start + chunk].to(lp.device).long()
            values = lp.gather(-1, selected)
        ids.append(selected.cpu().to(torch.int32))
        logps.append(values.cpu())
        if own_topk:
            own_values, own_selected = lp.topk(k, dim=-1)
            own_ids.append(own_selected.cpu().to(torch.int32))
            own_logps.append(own_values.cpu())
        if sampled is not None:
            target = torch.as_tensor(sampled[start:start + chunk], device=lp.device).long()
            sampled_lps.append(lp.gather(-1, target[:, None]).squeeze(-1).cpu())
        excluded_mass.append(raw_lp[:, support:].exp().sum(-1).cpu())
        undefined_mass.append(raw_lp[:, token_vocab_size:].exp().sum(-1).cpu())
    result = {"ids": torch.cat(ids), "logps": torch.cat(logps),
            "sampled_logps": torch.cat(sampled_lps) if sampled_lps else None,
            "undefined_mass": torch.cat(undefined_mass), "excluded_mass": torch.cat(excluded_mass)}
    if own_topk:
        result["own_ids"], result["own_logps"] = torch.cat(own_ids), torch.cat(own_logps)
    if include_entropy:
        result["entropy"] = torch.cat(entropies)
        result["entropy_support_size"] = support
    return result


def original_advantage(old, teacher):
    """dp_actor.py only_stu/student_p: normalize weights, NOT log-ratio."""
    return ((teacher - old) * old.softmax(-1)).detach()


def original_token_losses(current, old, advantage, clip=0.2, dual_clip=3.0):
    """Exact original 3D PPO branch, before the token-mean reduction."""
    ratio = (current - old).clamp(-20, 20).exp()
    first = -advantage * ratio
    clipped = -advantage * ratio.clamp(1 - clip, 1 + clip)
    maximum = torch.maximum(first, clipped)
    dual = torch.minimum(-advantage * dual_clip, maximum)
    return torch.where(advantage < 0, dual, maximum).sum(-1)


def response_loss(model, row, scores, controller, denominator, chunk=128, objective="reference_surrogate", loss_options=None):
    loss_options = loss_options or {}
    prefix, response = row["prompt_token_ids"], row["token_ids"]
    device = model.lm_head.weight.device
    tokens = torch.tensor([prefix + response[:-1]], dtype=torch.long, device=device)
    behavior_mask = row.get("steering_mask")
    learning_mask = behavior_mask
    if row.get("learning_steering") is False:
        learning_mask = [False] * len(response)
    elif row.get("steering_phase") == "learning_only":
        learning_mask = [True] * len(response)
    mask = response_prediction_mask(len(prefix), len(response), learning_mask, device=device)
    # Scope deliberately remains active THROUGH backward/checkpoint recompute.
    scope = controller.loss_scope(mask) if controller else nullcontext()
    with scope:
        hidden = model.model(input_ids=tokens, use_cache=False).last_hidden_state[0, len(prefix) - 1:]
        values = []
        extra_values = []
        for start in range(0, len(hidden), chunk):
            selected = scores["ids"][start:start + chunk].to(device).long()
            if objective in ("eopd_top16_surrogate", "eopd_sampled"):
                selected = torch.cat([selected, scores["teacher_topk_ids"][start:start + chunk].to(device).long()], -1)
            if objective in ("compass_sampled", "sampled_rkl", "eopd_sampled"):
                sampled = torch.tensor(row["token_ids"][start:start + chunk], device=device)[:, None]
                selected = torch.cat([selected, sampled], -1)

            def selected_logps(states, indices):
                return project_logits(model, states).log_softmax(-1).gather(-1, indices)

            value = checkpoint(selected_logps, hidden[start:start + chunk], selected, use_reentrant=False)
            values.append(value[:, :scores["ids"].shape[-1]])
            if objective in ("eopd_top16_surrogate", "compass_sampled", "sampled_rkl", "eopd_sampled"):
                extra_values.append(value[:, scores["ids"].shape[-1]:])
        current = torch.cat(values)
        old = scores["logps"].to(device)
        teacher = scores["teacher_logps"].to(device)
        advantage = original_advantage(old, teacher)
        # Original dp_actor on_policy=True sets old=current.detach(). At one
        # update per fresh batch this preserves its exact surrogate behavior.
        if objective in ("reference_surrogate", "eopd_top16_surrogate"):
            losses = original_token_losses(current, current.detach(), advantage)
            if objective == "eopd_top16_surrogate":
                teacher_top = scores["teacher_topk_logps"].to(device).log_softmax(-1)
                forward_kl = (teacher_top.exp() * (teacher_top - torch.cat(extra_values))).sum(-1)
                high_entropy = scores["teacher_entropy"].to(device) > loss_options.get("eopd_entropy_threshold", .8)
                losses = losses + loss_options.get("eopd_forward_kl_coefficient", 1.) * high_entropy * forward_kl
        elif objective == "candidate_rkl":
            # Explicit fixed-state, finite-candidate objective. No claim that
            # mismatched behavior and learning interventions are on-policy.
            logp, logq = current.log_softmax(-1), teacher.log_softmax(-1)
            losses = (logp.exp() * (logp - logq)).sum(-1)
        elif objective == "compass_candidate":
            logp = current.log_softmax(-1)
            reference = scores["reference_logps"].to(device)
            anchor = scores["anchor_logps"].to(device).log_softmax(-1)
            alpha = loss_options.get("compass_anchor_coefficient", .5)
            if alpha <= 0:
                raise ValueError("Compass anchor coefficient must be positive")
            losses = (logp.exp() * (alpha * (logp - anchor) - (teacher - reference))).sum(-1)
        elif objective == "compass_sampled":
            # Compass Eq.(9): SAME sampled action, raw full-support logps and
            # a frozen student initialization anchor. No candidate softmax.
            alpha = loss_options.get("compass_anchor_coefficient", .5)
            signal = (scores["teacher_sampled_logps"].to(device) - scores["reference_sampled_logps"].to(device)
                      - alpha * (scores["sampled_logps"].to(device)-scores["anchor_sampled_logps"].to(device))).detach()
            sampled_current = torch.cat(extra_values)
            losses = original_token_losses(sampled_current, sampled_current.detach(), signal[:, None])
        elif objective in ("sampled_rkl", "eopd_sampled"):
            extra = torch.cat(extra_values)
            sampled_current = extra[:, -1:]
            signal = (scores["teacher_sampled_logps"].to(device)-scores["sampled_logps"].to(device)).detach()
            losses = original_token_losses(sampled_current, sampled_current.detach(), signal[:, None])
            if objective == "eopd_sampled":
                logq = scores["teacher_topk_logps"].to(device).log_softmax(-1)
                fkl = (logq.exp() * (logq-extra[:, :-1])).sum(-1)
                high = scores["teacher_entropy"].to(device) > loss_options.get("eopd_entropy_threshold", .8)
                losses = losses + loss_options.get("eopd_forward_kl_coefficient", 1.) * high * fkl
        else:
            raise ValueError(f"Unsupported loss objective: {objective}")
        loss = losses.sum() / denominator
        replay_error = (current.detach() - old).abs().max().item()
        matching_policy = learning_mask == behavior_mask
        if scores.get("source") in ("vllm_rollout_raw_logprobs", "hf_controlled_rollout_raw_logprobs") and matching_policy:
            from steer_opd.replay_validation import verify_rollout_topk_replay
            scores["replay_validation"] = verify_rollout_topk_replay(old, current)
        elif not matching_policy:
            if objective != "candidate_rkl":
                raise ValueError("Mismatched intervention requires candidate_rkl")
            scores["replay_validation"] = {"matching_behavior_policy": False,
                "objective": objective, "expected_distribution_change": True}
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite OPD loss")
        loss.backward()
    return float(loss.detach()), replay_error


class MasterAdam:
    """BF16 forward weights with FP32 gradients, master weights and Adam state."""

    def __init__(self, model, config):
        self.named = list(model.named_parameters())
        self.master = [torch.nn.Parameter(p.detach().float().clone()) for _, p in self.named]
        self.optimizer = torch.optim.AdamW(
            self.master, lr=config["learning_rate"], betas=tuple(config["betas"]),
            eps=config.get("adam_eps", 1e-8), weight_decay=config["weight_decay"], foreach=False,
        )

    def accumulate(self):
        for (_, param), master in zip(self.named, self.master):
            if param.grad is not None:
                if master.grad is None:
                    master.grad = param.grad.float()
                else:
                    master.grad.add_(param.grad.float())
                param.grad = None

    def step(self, max_norm):
        norm = torch.nn.utils.clip_grad_norm_(self.master, max_norm, error_if_nonfinite=True)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            for (_, param), master in zip(self.named, self.master):
                param.copy_(master)
        return float(norm)

    def state_dict(self):
        return {"names": [n for n, _ in self.named],
                "master": [p.detach().cpu() for p in self.master],
                "optimizer": self.optimizer.state_dict(),
                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                "python_rng": random.getstate(), "numpy_rng": np.random.get_state()}

    def load(self, state):
        if state["names"] != [n for n, _ in self.named]:
            raise ValueError("Checkpoint parameter order mismatch")
        with torch.no_grad():
            for (_, param), master, saved in zip(self.named, self.master, state["master"]):
                master.copy_(saved)
                param.copy_(master)
        self.optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
