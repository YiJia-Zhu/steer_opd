"""Four S0 banks from one common T/Ref/S text-and-activation pool."""

import torch

from steer_opd.direction_bank import make_bank
from steer_opd.method_config import Method


def make_common_banks(pooled, layers, identity, *, alpha=.1):
    valid = {qid: record for qid, record in pooled.items() if record["usable"]}
    if not valid:
        raise ValueError("No usable common T/Ref/S activation triplets")
    scales = {str(layer): torch.cat([norm.float() for record in valid.values()
              for norm in record["layers"][str(layer)]["norms"]]).median() for layer in layers}
    result = {}
    for direction, negative_index in [("T-S", 2), ("T-R", 1)]:
        pairs = {}
        for qid, record in pooled.items():
            if not record["usable"]:
                pairs[qid] = record
                continue
            pairs[qid] = {"usable": True, "layers": {str(layer): {
                "pools": [record["layers"][str(layer)]["pools"][0], record["layers"][str(layer)]["pools"][negative_index]],
                "norms": record["layers"][str(layer)]["norms"]} for layer in layers}}
        for granularity in ("global", "per_question"):
            method = Method(direction=direction, granularity=granularity, extractor="initial", layers=tuple(layers),
                            gate="all", operator="slerp", alpha=alpha)
            name = f"{direction}_{granularity}"
            result[name] = make_bank(pairs, method, {**identity, "zero_direction_policy": "exact_no_intervention"},
                                     fixed_scales=scales, allow_zero_directions=True)
    assert len({tuple(bank["valid_ids"]) for bank in result.values()}) == 1
    return result, scales
