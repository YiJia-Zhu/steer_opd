"""Distribution checks for short replies with noisy sampled-token errors."""

import torch


def verify_rollout_topk_replay(rollout_logps, replay_logps, mean_limit=0.005, max_limit=0.03):
    """Compare the observed K events and their complement, plus conditional K.

    These are observable reduced distributions, NOT full-vocabulary KL. Real
    full-support backend acceptance is still required before a new campaign.
    """
    if rollout_logps.shape != replay_logps.shape or rollout_logps.ndim != 2 or not rollout_logps.numel():
        raise RuntimeError("Rollout Top-K replay shape mismatch")
    p, q = rollout_logps.detach().double(), replay_logps.detach().double()
    if (not torch.isfinite(p).all() or not torch.isfinite(q).all()
            or (p > 1e-6).any() or (q > 1e-6).any()
            or (p.exp().sum(-1) > 1 + 1e-5).any() or (q.exp().sum(-1) > 1 + 1e-5).any()):
        raise RuntimeError("Invalid rollout Top-K replay probabilities")

    def partition(lp):
        tail = (1 - lp.exp().sum(-1, keepdim=True)).clamp_min(1e-12).log()
        result = torch.cat([lp, tail], -1)
        return result - result.logsumexp(-1, keepdim=True)

    def symmetric_kl(a, b):
        return ((a.exp() * (a - b)).sum(-1).clamp_min(0),
                (b.exp() * (b - a)).sum(-1).clamp_min(0))

    coarse = symmetric_kl(partition(p), partition(q))
    conditional = symmetric_kl(p.log_softmax(-1), q.log_softmax(-1))
    metrics = {}
    for name, pair in [("topk_plus_complement", coarse), ("conditional_topk", conditional)]:
        for direction, value in zip(["forward", "reverse"], pair):
            metrics[f"{name}_{direction}_mean_kl"] = float(value.mean())
            metrics[f"{name}_{direction}_max_kl"] = float(value.max())
    passed = all(float(v.mean()) < mean_limit and float(v.max()) < max_limit
                 for v in (*coarse, *conditional))
    report = {"method": "rollout_topk_plus_complement_and_conditional_kl", "passed": passed,
              "full_vocabulary_kl": False, "positions": len(p),
              "mean_logp_error": float((p - q).abs().mean()),
              "max_logp_error": float((p - q).abs().max()),
              "threshold_mean_kl": mean_limit, "threshold_max_kl": max_limit, **metrics}
    if not passed:
        raise RuntimeError(f"Rollout Top-K/HF replay mismatch: {report}")
    return report


def verify_short_replay(captured_logits, replay_logits, token_ids, rollout_logps):
    """Use the existing full-distribution acceptance limits on every token.

    Captures must come from the actual rollout, not a separately resampled reply.
    The sampled-logprob check binds those captures to the recorded trajectory.
    """
    if captured_logits.shape != replay_logits.shape or captured_logits.ndim != 2:
        raise RuntimeError("Replay distribution shape mismatch")
    if not 1 <= len(token_ids) <= 8 or len(token_ids) != len(captured_logits):
        raise RuntimeError("Full-distribution fallback is only for complete short replies")
    reference = captured_logits.float().log_softmax(-1)
    replay = replay_logits.float().log_softmax(-1)
    ids = torch.as_tensor(token_ids, device=reference.device).long()
    recorded = torch.as_tensor(rollout_logps, device=reference.device).float()
    capture_error = (reference.gather(-1, ids[:, None]).squeeze(-1) - recorded).abs().max()
    kls = (reference.exp() * (reference - replay)).sum(-1)
    reverse = (replay.exp() * (replay - reference)).sum(-1)
    finite = torch.isfinite(reference).all() & torch.isfinite(replay).all()
    passed = bool(finite and capture_error < 1e-4 and kls.mean() < 0.005 and kls.max() < 0.03
                  and reverse.mean() < 0.005 and reverse.max() < 0.03)
    report = {"method": "actual_rollout_full_distribution", "passed": passed,
              "tokens": len(token_ids), "capture_sampled_logp_max_error": float(capture_error),
              "forward_kls": kls.tolist(), "reverse_kls": reverse.tolist(),
              "threshold_mean_kl": 0.005, "threshold_max_kl": 0.03}
    if not passed:
        raise RuntimeError(f"Rollout/HF full-distribution mismatch: {report}")
    return report
