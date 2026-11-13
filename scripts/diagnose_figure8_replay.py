#!/usr/bin/env python3
"""Read-only replay diagnostics; never commits scores or optimizer updates."""

import argparse
import os
from pathlib import Path

import torch

from steer_opd.resumable_core import (
    ROOT, atomic_json, controller_for, load_model, project_logits, read_json,
    response_hidden, score_hidden,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--key")
    args = parser.parse_args()
    job = read_json(args.job)
    os.environ["STEER_OPD_SUPPORT_SIZE"] = str(job["support_size"])
    model = load_model(ROOT / job["model"])
    hook = controller_for(model, ROOT / job["vector"] if job.get("vector") else None)
    reports = []
    with torch.inference_mode():
        for request in job["requests"]:
            key = request["key"]
            if args.key and args.key != key:
                continue
            row = read_json(ROOT / job["output"] / f"{key}.json")
            hidden = response_hidden(model, row, hook)
            score = score_hidden(model, hidden, sampled=row["token_ids"])
            original = torch.tensor(row["sampled_logps"])
            error = (score["sampled_logps"] - original).abs()
            report = {"key": key, "tokens": len(row["token_ids"]),
                      "mean_error": error.mean().item(), "max_error": error.max().item()}
            if len(error) <= 8:
                report.update({"vllm_logps": original.tolist(),
                               "hf_logps": score["sampled_logps"].tolist(),
                               "token_ids": row["token_ids"]})
            captures = ROOT / job["output"] / "full_logits"
            kls = []
            for position in [0, 1, 7, 15, 31]:
                path = captures / f"{row.get('capture_request_id')}_{position}.pt"
                if position >= len(hidden) or not path.exists():
                    continue
                v_lp = torch.load(path, weights_only=True).cuda().float().log_softmax(-1)
                h_lp = project_logits(model, hidden[position:position + 1])[0].log_softmax(-1)
                kls.append({"position": position,
                            "vllm_to_hf": (v_lp.exp() * (v_lp - h_lp)).sum().item(),
                            "hf_to_vllm": (h_lp.exp() * (h_lp - v_lp)).sum().item()})
            if kls:
                report["full_vocab_kls"] = kls
            reports.append(report)
    summary = {"rows": len(reports), "job": args.job, "reports": reports,
               "row_mean_error": sum(r["mean_error"] for r in reports) / len(reports),
               "token_mean_error": sum(r["tokens"] * r["mean_error"] for r in reports) /
                                   sum(r["tokens"] for r in reports),
               "failed_keys": [r["key"] for r in reports if r["mean_error"] > 0.12]}
    atomic_json(args.output, summary)
    print({k: v for k, v in summary.items() if k != "reports"}, flush=True)
    for report in reports:
        if report["mean_error"] > 0.06 or report.get("full_vocab_kls"):
            print(report, flush=True)


if __name__ == "__main__":
    main()
