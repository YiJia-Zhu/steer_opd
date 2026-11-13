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

from steer_opd.steering import SteeringController, find_decoder_layer


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


def controller_for(model, vector_path):
    if not vector_path:
        return None
    payload = torch.load(vector_path, map_location="cpu", weights_only=True)
    return SteeringController(find_decoder_layer(model, payload["layer"]), payload["delta"], 1.0)


def response_hidden(model, row, controller=None):
    """Return states predicting every actual response token, including EOS."""
    prefix, response = row["prompt_token_ids"], row["token_ids"]
    if not prefix or not response:
        raise ValueError("Empty prompt/response")
    tokens = torch.tensor([prefix + response[:-1]], dtype=torch.long, device="cuda")
    mask = torch.zeros_like(tokens, dtype=torch.bool)
    mask[:, len(prefix) - 1:] = True
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
    if not full_vocab:
        weight = weight[:getattr(model, "_opd_vocab_size", weight.shape[0])]
    if torch.is_grad_enabled() and (flat.requires_grad or weight.requires_grad):
        logits = FP32OutputLinear.apply(flat, weight)
    elif flat.is_cuda and flat.dtype == torch.bfloat16:
        logits = torch.mm(flat, weight.t(), out_dtype=torch.float32)
    else:
        logits = torch.nn.functional.linear(flat.float(), weight.float())
    return logits.reshape(*hidden.shape[:-1], -1)


def score_hidden(model, hidden, k=16, candidates=None, chunk=128, sampled=None):
    ids, logps, sampled_lps, undefined_mass, excluded_mass = [], [], [], [], []
    token_vocab_size = getattr(model, "_defined_vocab_size", model.config.vocab_size)
    for start in range(0, len(hidden), chunk):
        logits = project_logits(model, hidden[start:start + chunk], full_vocab=True)
        support = getattr(model, "_opd_vocab_size", logits.shape[-1])
        raw_lp = logits.log_softmax(-1)
        lp = logits[:, :support].log_softmax(-1)
        if candidates is None:
            values, selected = lp.topk(k, dim=-1)
        else:
            selected = candidates[start:start + chunk].to(lp.device).long()
            values = lp.gather(-1, selected)
        ids.append(selected.cpu().to(torch.int32))
        logps.append(values.cpu())
        if sampled is not None:
            target = torch.as_tensor(sampled[start:start + chunk], device=lp.device).long()
            sampled_lps.append(lp.gather(-1, target[:, None]).squeeze(-1).cpu())
        excluded_mass.append(raw_lp[:, support:].exp().sum(-1).cpu())
        undefined_mass.append(raw_lp[:, token_vocab_size:].exp().sum(-1).cpu())
    return {"ids": torch.cat(ids), "logps": torch.cat(logps),
            "sampled_logps": torch.cat(sampled_lps) if sampled_lps else None,
            "undefined_mass": torch.cat(undefined_mass), "excluded_mass": torch.cat(excluded_mass)}


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


def response_loss(model, row, scores, controller, denominator, chunk=128):
    prefix, response = row["prompt_token_ids"], row["token_ids"]
    tokens = torch.tensor([prefix + response[:-1]], dtype=torch.long, device="cuda")
    mask = torch.zeros_like(tokens, dtype=torch.bool)
    mask[:, len(prefix) - 1:] = True
    # Scope deliberately remains active THROUGH backward/checkpoint recompute.
    scope = controller.loss_scope(mask) if controller else nullcontext()
    with scope:
        hidden = model.model(input_ids=tokens, use_cache=False).last_hidden_state[0, len(prefix) - 1:]
        values = []
        for start in range(0, len(hidden), chunk):
            selected = scores["ids"][start:start + chunk].to("cuda").long()

            def selected_logps(states, indices):
                return project_logits(model, states).log_softmax(-1).gather(-1, indices)

            values.append(checkpoint(selected_logps, hidden[start:start + chunk], selected,
                                     use_reentrant=False))
        current = torch.cat(values)
        old = scores["logps"].to("cuda")
        teacher = scores["teacher_logps"].to("cuda")
        advantage = original_advantage(old, teacher)
        # Original dp_actor on_policy=True sets old=current.detach(). At one
        # update per fresh batch this preserves its exact surrogate behavior.
        losses = original_token_losses(current, current.detach(), advantage)
        loss = losses.sum() / denominator
        replay_error = (current.detach() - old).abs().max().item()
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
