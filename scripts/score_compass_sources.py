#!/usr/bin/env python3
"""Batch-score native T/Ref tokenizations of the completed student response."""

import argparse

import torch

from steer_opd.compass_job import prepare_plans, project_path, model_path
from steer_opd.compass_scoring import load_text_scorer
from steer_opd.execution_policy import require_execution_allowed
from steer_opd.resumable_core import atomic_torch, digest, read_json
from steer_opd.static_screen import score_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True)
    parser.add_argument("--role", choices=("teacher", "reference"), required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    job = read_json(args.job)
    if args.execute:
        require_execution_allowed(job)
    if job.get("optimizer_updates") != 0:
        raise ValueError("Source scoring must not update model parameters")
    _, _, role_rows, codecs = prepare_plans(job)
    if not args.execute:
        print(f"CPU alignment complete; {len(role_rows[args.role])} {args.role} prefixes prepared")
        return
    codec = codecs[args.role]
    output = project_path(job[f"{args.role}_scores"])
    pending = []
    for row in role_rows[args.role]:
        binding = digest({"row": row, "checkpoint": job[args.role], "tokenizer": codec.mapping_sha256,
                          "support_id": codec.support_id})
        p = output / f"{row['key']}.pt"
        if p.exists():
            if torch.load(p, weights_only=True, map_location="cpu").get("binding") != binding:
                raise ValueError("Native scorer cache identity changed")
        else:
            pending.append((row, p, binding))
    if not pending:
        print("All native source caches complete")
        return
    batch_size = job.get(f"{args.role}_batch_size", 1)
    if batch_size < 1:
        raise ValueError("Invalid source scoring batch size")
    model = load_text_scorer(model_path(job[args.role]), codec)
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start+batch_size]
        scores = score_batch(model, [row for row, _, _ in batch], chunk=job.get("chunk", 128))
        for (row, p, binding), score in zip(batch, scores):
            atomic_torch(p, {**score, "binding": binding, "alignment_id": row["alignment_id"],
                "role": args.role, "checkpoint": job[args.role], "support_size": codec.support_size,
                "tokenizer_mapping_sha256": codec.mapping_sha256})


if __name__ == "__main__":
    main()
