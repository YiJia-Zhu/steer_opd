#!/usr/bin/env python3
"""Batch-score a completed unsteered source once; no online gate or student pass."""

import argparse
import os

import torch

from steer_opd.resumable_core import atomic_torch, digest, load_model, read_json, require_gpu
from steer_opd.static_masks import source_binding
from steer_opd.static_screen import score_batch
from run_static_overlap_screen import path
from steer_opd.compass_job import model_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    job = read_json(args.job)
    rows = read_json(path(job["rows"]))
    if job.get("optimizer_updates") != 0 or job["identity"]["metric"] != "native_overlap_at_16":
        raise ValueError("This native teacher scorer needs an explicit zero-update, same-tokenizer protocol")
    if job["identity"]["student_tokenizer_sha256"] != job["identity"]["teacher_tokenizer_sha256"]:
        raise ValueError("Cross-tokenizer teacher scoring requires an accepted same-text event adapter")
    if job["teacher"] != job["identity"]["teacher_checkpoint"]:
        raise ValueError("Teacher checkpoint differs from the source identity")
    output = path(job["teacher_scores"])
    pending = []
    for row in rows:
        identity = {**job["identity"], "student_score_sha256": "teacher-preflight", "teacher_score_sha256": "teacher-preflight"}
        source_binding(row, identity)  # Reject controlled histories before GPU work.
        key = row.get("key", row["id"])
        if path(key).name != key or key in (".", ".."):
            raise ValueError("Unsafe response key")
        binding = digest({"row": row, "identity": job["identity"], "support_size": job["support_size"]})
        target = output / f"{key}.pt"
        if target.exists():
            if torch.load(target, map_location="cpu", weights_only=True).get("binding") != binding:
                raise ValueError("Teacher source cache changed")
        else:
            pending.append((row, target, binding))
    if not pending:
        print("All teacher source caches already complete")
        return
    require_gpu()
    os.environ["STEER_OPD_SUPPORT_SIZE"] = str(job["support_size"])
    model = load_model(model_path(job["teacher"]))
    batch_size = job.get("teacher_batch_size", 4)
    if batch_size < 1:
        raise ValueError("Invalid teacher batch size")
    for begin in range(0, len(pending), batch_size):
        group = pending[begin:begin+batch_size]
        scores = score_batch(model, [row for row, _, _ in group], chunk=job.get("chunk", 128))
        for (row, target, binding), score in zip(group, scores):
            atomic_torch(target, {**score, "binding": binding, "support_size": job["support_size"],
                "teacher_checkpoint": job["identity"]["teacher_checkpoint"],
                "source_token_hash": digest({"prompt_token_ids": row["prompt_token_ids"], "token_ids": row["token_ids"]})})


if __name__ == "__main__":
    main()
