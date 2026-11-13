"""Causal random controls and training/dev-only gate calibration."""

import math
import random


def matched_draw(history, shadow, position, draw, initial_rate, bin_width, position_matched):
    """Choose using STRICTLY earlier shadow decisions, then record this step.

    The shadow policy is evaluated on this control's own visited states. This
    does not guarantee exact count equality to another free-running policy.
    """
    key = str(position // bin_width) if position_matched else "all"
    counts = history.setdefault(key, [0, 0])
    probability = counts[1] / counts[0] if counts[0] else initial_rate
    inject = draw < probability
    counts[0] += 1
    counts[1] += int(shadow)
    return inject, {"matching_probability": probability, "shadow_gate": bool(shadow),
                    "matching_bin": key, "matching_uses_current_shadow": False}


def offline_matched_mask(mask, seed, bin_width=None):
    """Exact count matching is valid only for a COMPLETE FIXED prefix."""
    rng = random.Random(seed)
    result = [False] * len(mask)
    width = bin_width or max(1, len(mask))
    for start in range(0, len(mask), width):
        stop = min(start + width, len(mask))
        count = sum(mask[start:stop])
        for position in rng.sample(range(start, stop), count):
            result[position] = True
    return result


def calibrate_threshold(values, target_rate, comparison, *, lower_bound=None):
    values = [float(value) for value in values]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("Calibration needs finite training/dev observations")
    if not 0 < target_rate < 1 or comparison not in ("lt", "gt"):
        raise ValueError("Invalid gate calibration")
    candidates = set(values)
    candidates.update(math.nextafter(value, math.inf if comparison == "lt" else -math.inf) for value in values)
    if lower_bound is not None:
        candidates = {max(lower_bound, value) for value in candidates}
    def rate(threshold):
        return sum(value < threshold if comparison == "lt" else value > threshold for value in values) / len(values)
    threshold = min(candidates, key=lambda value: (abs(rate(value) - target_rate), value))
    return {"threshold": threshold, "target_rate": target_rate, "observed_rate": rate(threshold),
            "positions": len(values), "comparison": comparison}
