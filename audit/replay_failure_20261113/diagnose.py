"""Read-only diagnosis of the retained first-batch trajectories; no updates."""
import argparse
import ast
import os
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from steer_opd.resumable_core import (ROOT, atomic_json, atomic_torch, controller_for,
    load_model, project_logits, read_json, response_hidden, score_hidden, emit)
from steer_opd.replay_validation import verify_rollout_topk_replay
from steer_opd.rollout_topk import load_rollout_topk
from steer_opd.steering import response_prediction_mask


def comparison(old, new):
    try:
        report = verify_rollout_topk_replay(old, new)
    except RuntimeError as error:
        if "mismatch: " not in str(error):
            raise
        report = ast.literal_eval(str(error).split("mismatch: ", 1)[1])
    p, q = old.double(), new.double()
    a, b = p.log_softmax(-1), q.log_softmax(-1)
    per_position = (a.exp() * (a - b)).sum(-1).clamp_min(0)
    values, indices = per_position.topk(min(8, len(per_position)))
    report["worst_positions"] = [{"position": int(i), "conditional_forward_kl": float(v)}
                                 for i, v in zip(indices, values)]
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--keys", nargs="*")
    parser.add_argument("--focus", action="store_true")
    args = parser.parse_args()
    job = read_json(args.job)
    os.environ["STEER_OPD_SUPPORT_SIZE"] = str(job["support_size"])
    model = load_model(ROOT / job["model"])
    output = ROOT / args.output
    reports = []
    started = time.monotonic()
    with torch.inference_mode():
        for request in job["requests"]:
            key = request["key"]
            if args.keys and key not in args.keys:
                continue
            row = read_json(ROOT / job["output"] / f"{key}.json")
            old = load_rollout_topk(ROOT / job["output"], row, job["save_rollout_top_k"], job["support_size"])
            controller = controller_for(model, ROOT / row["vector_path"], row)
            try:
                hidden = response_hidden(model, row, controller)
                new = score_hidden(model, hidden, candidates=old["ids"])
                report = {"key": key, "tokens": len(row["token_ids"]), **comparison(old["logps"], new["logps"])}
                atomic_torch(output / f"{key}_parallel.pt", {"logps": new["logps"], "ids": old["ids"]})
                if args.focus:
                    target = report["worst_positions"][0]["position"]
                    prefix = row["prompt_token_ids"] + row["token_ids"][:target]
                    tokens = torch.tensor([prefix], device="cuda")
                    mask = response_prediction_mask(len(row["prompt_token_ids"]), target + 1,
                                                     row["steering_mask"][:target+1], device="cuda")
                    variants = []
                    for backend in [None, SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]:
                        scope = sdpa_kernel(backend) if backend is not None else nullcontext()
                        with scope:
                            with controller.loss_scope(mask):
                                h = model.model(input_ids=tokens, use_cache=False).last_hidden_state[:, -1]
                            lp = project_logits(model, h).log_softmax(-1)
                        selected = lp.gather(-1, old["ids"][target:target+1].long().cuda()).cpu()
                        variants.append({"backend": str(backend), "position": target,
                                         **comparison(old["logps"][target:target+1], selected)})
                        atomic_torch(output / f"{key}_{backend}_position{target}.pt", {"logps": lp.cpu()})
                        scope = sdpa_kernel(backend) if backend is not None else nullcontext()
                        with scope:
                            with controller.loss_scope(mask[:, :-1]):
                                cache = model.model(input_ids=tokens[:, :-1], use_cache=True).past_key_values
                            with controller.loss_scope(mask[:, -1:]):
                                h = model.model(input_ids=tokens[:, -1:], past_key_values=cache,
                                                use_cache=True).last_hidden_state[:, -1]
                            lp = project_logits(model, h).log_softmax(-1)
                            del cache
                        selected = lp.gather(-1, old["ids"][target:target+1].long().cuda()).cpu()
                        variants.append({"backend": str(backend), "cached_last": True, "position": target,
                                         **comparison(old["logps"][target:target+1], selected)})
                    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
                    with controller.loss_scope(mask):
                        h = model.model(input_ids=tokens, use_cache=False).last_hidden_state[:, -1]
                    lp = project_logits(model, h).log_softmax(-1)
                    selected = lp.gather(-1, old["ids"][target:target+1].long().cuda()).cpu()
                    variants.append({"backend": "no_bf16_reduced_precision_reduction", "position": target,
                                     **comparison(old["logps"][target:target+1], selected)})
                    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
                    report["prefix_variants"] = variants
            finally:
                controller.detach()
            reports.append(report)
            emit("diagnostic_row", key=key, tokens=report["tokens"], passed=report["passed"],
                 max_kl=max(v for k,v in report.items() if k.endswith("max_kl")),
                 worst=report["worst_positions"][:2])
            atomic_json(output / "report.json", {"job": args.job, "experimental_updates": 0,
                "elapsed_seconds": time.monotonic()-started, "rows": reports,
                "failed_keys": [r["key"] for r in reports if not r["passed"]]})


if __name__ == "__main__":
    main()
