"""Paired fixed-prefix metric summaries and select-only threshold calibration."""
import math
from bisect import bisect_left, bisect_right
from collections import Counter
from itertools import accumulate
import numpy as np
import torch


def position_metrics(student, teacher):
    if (student["ids"].shape != teacher["ids"].shape or student["logps"].shape != student["ids"].shape
            or teacher["logps"].shape != teacher["ids"].shape):
        raise ValueError("Overlap diagnostic needs identical token positions and K")
    matches = student["ids"][:, :, None] == teacher["ids"][:, None, :]
    sm, tm = matches.any(-1), matches.any(-2)
    return {"overlap": sm.float().mean(-1),
            "student_shared_mass": (student["logps"].exp() * sm).sum(-1),
            "teacher_shared_mass": (teacher["logps"].exp() * tm).sum(-1)}


def paired_summary(records, baseline, seed=42, bootstrap_samples=2000):
    ids = sorted(records)
    deltas = np.array([float(records[q]["overlap"].mean() - baseline[q]["overlap"].mean()) for q in ids])
    lengths = np.array([len(records[q]["overlap"]) for q in ids])
    candidate_sum = sum(float(records[q]["overlap"].sum()) for q in ids)
    baseline_sum = sum(float(baseline[q]["overlap"].sum()) for q in ids)
    rng = np.random.default_rng(seed)
    bootstrap = deltas[rng.integers(0, len(ids), size=(bootstrap_samples, len(ids)))].mean(-1)
    buckets = {}
    for label, lower, upper in [("0_512", 0, 512), ("513_2048", 513, 2048),
                                ("2049_4096", 2049, 4096), ("4097_plus", 4097, float("inf"))]:
        chosen = (lengths >= lower) & (lengths <= upper)
        if chosen.any():
            buckets[label] = {"questions": int(chosen.sum()), "question_weighted_delta_pp": float(deltas[chosen].mean() * 100)}
    return {"questions": len(ids), "positions": int(lengths.sum()), "length_buckets": buckets,
            "question_weighted_delta_pp": float(deltas.mean() * 100),
            "paired_bootstrap_95ci_pp": (np.quantile(bootstrap, [0.025, 0.975]) * 100).tolist(),
            "token_weighted_overlap16": candidate_sum / int(lengths.sum()),
            "token_weighted_delta_pp": 100 * (candidate_sum - baseline_sum) / int(lengths.sum()),
            "improved_questions": int((deltas > 0).sum()),
            "student_shared_mass": float(np.mean([float(records[q]["student_shared_mass"].mean()) for q in ids])),
            "teacher_shared_mass": float(np.mean([float(records[q]["teacher_shared_mass"].mean()) for q in ids])),
            "per_question_delta_pp": {qid: float(delta * 100) for qid, delta in zip(ids, deltas)},
            "parameter_updates": 0, "interpretation": "immediate_fixed_prefix_intervention_not_learning"}


def calibrate_threshold(values, target_rate, comparison, *, lower_bound=None):
    """Count each score once; Top16 count-overlap has at most 17 bins.

    Keep the original strict comparison, nextafter candidates and lower-threshold
    tie break. Continuous metrics also avoid rescanning all N observations for
    every candidate: O(N + U log U), where U is the number of distinct scores.
    """
    counts = Counter()
    for value in values:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("Calibration needs finite training/dev observations")
        counts[value] += 1
    if not counts:
        raise ValueError("Calibration needs finite training/dev observations")
    if not 0 < target_rate < 1 or comparison not in ("lt", "gt"):
        raise ValueError("Invalid gate calibration")
    scores = sorted(counts)
    cumulative = [0, *accumulate(counts[value] for value in scores)]
    total = cumulative[-1]
    candidates = set(scores)
    candidates.update(math.nextafter(value, math.inf if comparison == "lt" else -math.inf) for value in scores)
    if lower_bound is not None:
        candidates = {max(lower_bound, value) for value in candidates}
    def rate(threshold):
        selected = (cumulative[bisect_left(scores, threshold)] if comparison == "lt" else
                    total-cumulative[bisect_right(scores, threshold)])
        return selected / total
    threshold = min(candidates, key=lambda value: (abs(rate(value) - target_rate), value))
    return {"threshold": threshold, "target_rate": target_rate, "observed_rate": rate(threshold),
            "positions": total, "comparison": comparison}
