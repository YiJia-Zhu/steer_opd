"""Shared tensor math for HF and engine steering; no scheduler state."""

import math

import torch


def validate_operator(operator, strength):
    if operator not in ("additive", "slerp"):
        raise ValueError("Unknown steering operator")
    if isinstance(strength, bool) or not math.isfinite(strength) or strength < 0 or (operator == "slerp" and strength > 1):
        raise ValueError("Slerp alpha must be in [0,1]; additive strength must be nonnegative")


def slerp(hidden, direction, alpha):
    """Rotate by alpha of the angle to v, preserving h's norm.

    Zero h/v and alpha=0 are exact no-ops. At the antipode use the
    least-aligned coordinate axis as a deterministic great-circle tangent.
    direction is [H] or broadcastable, e.g. [B,1,H].
    """
    validate_operator("slerp", alpha)
    if alpha == 0:
        return hidden
    if hidden.shape[-1] < 2:
        raise ValueError("Slerp needs at least two hidden dimensions")
    dtype = torch.float64 if hidden.dtype == torch.float64 else torch.float32
    h = hidden.to(dtype)
    v = direction.to(device=hidden.device, dtype=dtype)
    hn, vn = h.norm(dim=-1, keepdim=True), v.norm(dim=-1, keepdim=True)
    eps = torch.finfo(dtype).eps
    valid = (hn > eps) & (vn > eps)
    u, target = h / hn.clamp_min(eps), v / vn.clamp_min(eps)
    dot = (u * target).sum(-1, keepdim=True).clamp(-1, 1)
    # Avoid undefined atan2(0,0) gradients, including unselected branches.
    dot = torch.where(valid, dot, torch.ones_like(dot))
    tangent = target - dot * u
    tangent_norm = tangent.norm(dim=-1, keepdim=True)
    basis = torch.nn.functional.one_hot(u.abs().argmin(-1), h.shape[-1]).to(dtype)
    fallback = basis - (basis * u).sum(-1, keepdim=True) * u
    fallback = fallback / fallback.norm(dim=-1, keepdim=True).clamp_min(eps)
    tangent_unit = torch.where(tangent_norm > eps, tangent / tangent_norm.clamp_min(eps), fallback)
    angle = torch.atan2(tangent_norm, dot) * alpha
    rotated = (angle.cos() * u + angle.sin() * tangent_unit) * hn
    rotated = rotated * hn / rotated.norm(dim=-1, keepdim=True).clamp_min(eps)
    unchanged = ~valid | ((tangent_norm <= eps) & (dot >= 0))
    return torch.where(unchanged, h, rotated).to(hidden.dtype)


def transform(hidden, vector, strength=1.0, operator="additive"):
    validate_operator(operator, strength)
    if strength == 0:
        return hidden
    if operator == "slerp":
        return slerp(hidden, vector, strength)
    return hidden + vector.to(hidden) * strength
