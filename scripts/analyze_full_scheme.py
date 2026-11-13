#!/usr/bin/env python3
"""CPU analysis of preserved pre-injection probes and direction drift."""

import argparse
from collections import defaultdict

import torch

from steer_opd.resumable_core import ROOT, atomic_json, read_json


def analyze(cfg):
    run = ROOT / cfg["run_dir"]
    bins = defaultdict(lambda: {"positions": 0, "injected": 0, "overlap_sum": 0., "shadow": 0, "shadow_observed": 0})
    for group in cfg["groups"]:
        for path in sorted((run / group / "steps").glob("*/token_statistics/*.pt")):
            record = torch.load(path, weights_only=True)
            gate = record["steering_mask"]
            overlap = record.get("probe_overlap16")
            axes = {"absolute": torch.arange(len(gate))//256}
            if overlap is not None:
                axes["probe_overlap"] = (overlap*4).long().clamp_max(3)
            shadow = observed = None
            if cfg.get("steering_method", {}).get("gate") in ("matched_random", "matched_position_random"):
                row = read_json(path.parent.parent / "rollouts" / f"{path.stem}.json")
                metrics = row["gate_metrics"]
                shadow = torch.tensor([int(value.get("shadow_gate", False)) for value in metrics]).float()
                observed = torch.tensor([int("shadow_gate" in value) for value in metrics]).float()
            for axis, indices in axes.items():
                counts = torch.bincount(indices)
                injected = torch.bincount(indices, weights=gate.float())
                overlaps = torch.bincount(indices, weights=overlap) if overlap is not None else None
                shadows = torch.bincount(indices, weights=shadow) if shadow is not None else None
                observed_counts = torch.bincount(indices, weights=observed) if observed is not None else None
                for index in counts.nonzero().flatten().tolist():
                    bucket = bins[f"{group}/{axis}/{index}"]
                    bucket["positions"] += int(counts[index])
                    bucket["injected"] += int(injected[index])
                    if overlaps is not None:
                        bucket["overlap_sum"] += float(overlaps[index])
                    if shadows is not None:
                        bucket["shadow"] += int(shadows[index])
                        bucket["shadow_observed"] += int(observed_counts[index])
    for value in bins.values():
        value["injection_rate"] = value["injected"]/value["positions"]
        if value["shadow_observed"]:
            value["matching_count_error"] = value["injected"]-value["shadow"]
            value["matching_rate_error"] = value["injected"]/value["positions"]-value["shadow"]/value["shadow_observed"]
    drift = []
    initial = None
    for path in sorted((run / "directions").glob("*/vectors.pt")):
        bank = torch.load(path, weights_only=True)
        if initial is None:
            initial = bank
        angles = []
        for qid in set(bank["per_question"]) & set(initial["per_question"]):
            for layer in bank["per_question"][qid]:
                current = bank["per_question"][qid][layer]["direction"].float()
                start = initial["per_question"][qid][layer]["direction"].float()
                angles.append(float(torch.nn.functional.cosine_similarity(current, start, dim=0)))
        drift.append({"bank": str(path.relative_to(ROOT)), "extractor_step": bank["identity"]["step"],
            "mean_cosine_to_first_bank": sum(angles)/len(angles) if angles else None, "common_question_layers": len(angles)})
    atomic_json(run / "mechanism_analysis.json", {"position_and_probe_buckets": dict(bins), "direction_drift": drift,
        "probe_unavailable_is_not_replaced_with_steered_overlap": True, "matching_is_against_shadow_on_own_trajectory": True,
        "does_not_establish_causal_benefit": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    analyze(read_json(parser.parse_args().config))
