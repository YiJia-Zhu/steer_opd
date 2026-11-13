"""Persist raw rollout Top-K independently of the sampled-token extra entry."""

import hashlib
from pathlib import Path

import torch

from steer_opd.resumable_core import atomic_json, atomic_torch, digest


SOURCE = "vllm_rollout_raw_logprobs"
HF_SOURCE = "hf_controlled_rollout_raw_logprobs"
ROLLOUT_SOURCES = (SOURCE, HF_SOURCE)


def extract_topk(token_ids, steps, k, support_size):
    if not token_ids or steps is None or len(steps) != len(token_ids):
        raise ValueError("Missing or incomplete rollout logprobs")
    ids, logps, sampled = [], [], []
    for token, entries in zip(token_ids, steps):
        if token not in entries or len(entries) not in (k, k + 1):
            raise ValueError("Rollout must return Top-K plus the sampled token")
        # vLLM inserts the sampled token before its actual topk entries. When
        # outside Top-K it is an extra item, even if its rank ties the boundary.
        ranked = [(value.rank, tid, float(value.logprob)) for tid, value in entries.items()
                  if not (len(entries) == k + 1 and tid == token)]
        if any(rank is None for rank, _, _ in ranked):
            raise ValueError("Missing vLLM Top-K rank")
        ranked.sort(key=lambda item: item[0])
        if [rank for rank, _, _ in ranked] != list(range(1, k + 1)):
            raise ValueError("Incomplete vLLM Top-K ranks")
        ids.append([tid for _, tid, _ in ranked])
        logps.append([lp for _, _, lp in ranked])
        sampled.append(float(entries[token].logprob))
    result = {"ids": torch.tensor(ids, dtype=torch.int32),
              "logps": torch.tensor(logps, dtype=torch.float32),
              "sampled_logps": torch.tensor(sampled, dtype=torch.float32),
              "source": SOURCE, "support_size": support_size, "k": k,
              "probability_mode": "raw_logprobs"}
    validate_topk(result, token_ids, k, support_size)
    return result


def attach_token_entropy(score, token_ids, statistics):
    if (statistics["positions"] != list(range(len(token_ids)))
            or statistics["support_size"] != score["support_size"]
            or statistics.get("source") != "raw_full_support_logits"):
        raise ValueError("vLLM token entropy positions/support/source mismatch")
    score["student_entropy"] = torch.tensor(statistics["entropy"], dtype=torch.float32)
    score["entropy_support_size"] = statistics["support_size"]
    validate_topk(score, token_ids, score["k"], score["support_size"])


def validate_topk(score, token_ids, k, support_size):
    ids, logps, sampled = score["ids"], score["logps"], score["sampled_logps"]
    if (score.get("source") not in ROLLOUT_SOURCES or score.get("probability_mode") != "raw_logprobs"
            or score.get("k") != k or score.get("support_size") != support_size):
        raise ValueError("Rollout Top-K probability protocol mismatch")
    if (ids.shape != (len(token_ids), k) or logps.shape != ids.shape
            or sampled.shape != (len(token_ids),) or not len(token_ids)
            or ids.dtype != torch.int32 or logps.dtype != torch.float32
            or sampled.dtype != torch.float32):
        raise ValueError("Rollout Top-K tensor shape/dtype mismatch")
    if (ids < 0).any() or (ids >= support_size).any() or any(t < 0 or t >= support_size for t in token_ids):
        raise ValueError("Rollout Top-K token outside frozen support")
    if (ids.sort(-1).values.diff(dim=-1) == 0).any():
        raise ValueError("Duplicate rollout Top-K ID")
    if (not torch.isfinite(logps).all() or not torch.isfinite(sampled).all()
            or (logps > 1e-6).any() or (sampled > 1e-6).any()
            or (logps.diff(dim=-1) > 1e-6).any()
            or (logps.double().exp().sum(-1) > 1 + 1e-5).any()):
        raise ValueError("Invalid raw rollout Top-K probabilities")
    matches = ids == torch.tensor(token_ids)[:, None]
    if ((logps - sampled[:, None]).abs()[matches] > 1e-5).any():
        raise ValueError("Sampled token probability differs from Top-K")
    outside = ~matches.any(-1)
    if (sampled[outside] > logps[outside, -1] + 1e-5).any():
        raise ValueError("Sampled token incorrectly excluded from Top-K")
    if "teacher_logps" in score:
        for key in ("teacher_logps", "teacher_topk_logps"):
            if (score[key].shape != ids.shape or score[key].dtype != torch.float32
                    or not torch.isfinite(score[key]).all() or (score[key] > 1e-6).any()):
                raise ValueError("Invalid saved online teacher probabilities")
        teacher_ids = score["teacher_topk_ids"]
        if (teacher_ids.shape != ids.shape or teacher_ids.dtype != torch.int32
                or (teacher_ids < 0).any() or (teacher_ids >= support_size).any()
                or (teacher_ids.sort(-1).values.diff(dim=-1) == 0).any()):
            raise ValueError("Invalid saved online teacher Top-K IDs")
        if (score["teacher_topk_logps"].double().exp().sum(-1) > 1 + 1e-5).any():
            raise ValueError("Saved online teacher mass exceeds one")
    for key in ("student_entropy", "teacher_entropy", "probe_entropy"):
        if key in score:
            value = score[key]
            if (score.get("entropy_support_size") != support_size or value.shape != (len(token_ids),)
                    or not torch.isfinite(value).all() or (value < -1e-6).any()):
                raise ValueError("Invalid full-support token entropy")
    if "teacher_sampled_logps" in score:
        value = score["teacher_sampled_logps"]
        if value.shape != (len(token_ids),) or not torch.isfinite(value).all() or (value > 1e-6).any():
            raise ValueError("Invalid sampled-token teacher logprobs")


def tensor_hash(score):
    sha = hashlib.sha256()
    keys = ["ids", "logps", "sampled_logps"]
    keys += [key for key in ("teacher_logps", "teacher_topk_ids", "teacher_topk_logps",
                             "probe_topk_ids", "probe_topk_logps", "student_entropy", "teacher_entropy",
                             "probe_entropy", "teacher_sampled_logps", "full_rollout_logps",
                             "full_teacher_logps", "full_probe_logps") if key in score]
    for key in keys:
        tensor = score[key].cpu().contiguous()
        sha.update(f"{key}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
        sha.update(tensor.numpy().tobytes())
    return sha.hexdigest()


def commit_topk_response(output, row, score):
    """The response is the commit marker; its tensor artifact is durable first."""
    output = Path(output)
    validate_topk(score, row["token_ids"], score["k"], score["support_size"])
    row = {**row, "rollout_topk": f"topk/{row['key']}.pt", "rollout_topk_hash": tensor_hash(score)}
    score = {**score, "row_hash": digest(row)}
    atomic_torch(output / row["rollout_topk"], score)
    atomic_json(output / f"{row['key']}.json", row)
    return row


def load_rollout_topk(output, row, k, support_size):
    output = Path(output).resolve()
    path = (output / row.get("rollout_topk", "__missing__")).resolve()
    if not path.is_relative_to(output) or not path.is_file():
        raise ValueError("Missing or invalid committed rollout Top-K artifact")
    score = torch.load(path, map_location="cpu", weights_only=True)
    validate_topk(score, row["token_ids"], k, support_size)
    if score.get("row_hash") != digest(row) or row.get("rollout_topk_hash") != tensor_hash(score):
        raise ValueError("Rollout Top-K integrity/provenance mismatch")
    if not torch.allclose(score["sampled_logps"], torch.tensor(row["sampled_logps"]), atol=1e-6, rtol=0):
        raise ValueError("Rollout sampled logprobs changed")
    return score


def overlap_totals(student, teacher):
    """Token sums, suitable for exact token-weighted aggregation across replies."""
    if student["ids"].shape != teacher["ids"].shape:
        raise ValueError("Overlap needs identical positions and K")
    matches = student["ids"][:, :, None] == teacher["ids"][:, None, :]
    sm, tm = matches.any(-1), matches.any(-2)
    return {"positions": len(sm), "intersection": int(sm.sum()),
            "student_shared_mass_sum": float((student["logps"].double().exp() * sm).sum()),
            "teacher_shared_mass_sum": float((teacher["logps"].double().exp() * tm).sum())}
