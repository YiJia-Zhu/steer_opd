"""Atomic artifact IO and frozen HF reference scoring; no training loop."""
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import torch
from steer_opd.steering import response_prediction_mask
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


def require_gpu():
    if not os.environ.get("STEER_OPD_GPU_ADMISSION"):
        raise RuntimeError("Fresh idle-GPU admission is required")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("This backend must see exactly one admitted GPU")


def load_model(path):
    from transformers import AutoModelForCausalLM

    require_gpu()
    model = AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation=os.environ.get("STEER_OPD_HF_ATTENTION", "sdpa"), device_map="cuda:0",
    )
    model.config.use_cache = False
    model.requires_grad_(False).eval()
    return model


def response_hidden(model, row, controller=None):
    """Return states predicting every actual response token, including EOS."""
    prefix, response = row["prompt_token_ids"], row["token_ids"]
    if not prefix or not response:
        raise ValueError("Empty prompt/response")
    device = model.lm_head.weight.device
    tokens = torch.tensor([prefix + response[:-1]], dtype=torch.long, device=device)
    mask = response_prediction_mask(len(prefix), len(response), row.get("steering_mask"), device=device)
    scope = controller.loss_scope(mask) if controller else nullcontext()
    with scope:
        result = model.model(input_ids=tokens, use_cache=False).last_hidden_state
    return result[0, len(prefix) - 1:]


def project_logits(model, hidden, full_vocab=False):
    # Match vLLM head_dtype=float32: BF16 operands, FP32 accumulator/output.
    # Avoid BF16 rounding of large logits before log-softmax (RL consistency).
    flat = hidden.reshape(-1, hidden.shape[-1])
    weight = model.lm_head.weight
    if flat.device != weight.device:
        flat = flat.to(weight.device)
    if not full_vocab:
        weight = weight[:getattr(model, "_opd_vocab_size", weight.shape[0])]
    if flat.is_cuda and flat.dtype == torch.bfloat16:
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
