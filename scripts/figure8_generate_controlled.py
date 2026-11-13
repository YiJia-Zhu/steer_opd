#!/usr/bin/env python3
"""Causal gated rollout reference backend, teacher scores saved in the same pass."""

import argparse
import os
import torch
from transformers import AutoTokenizer

from steer_opd.controlled_rollout import generate_response
from steer_opd.generation_runtime import prepare_generation_manifest
from steer_opd.method_config import Method
from steer_opd.direction_bank import validate_request_vectors
from steer_opd.resumable_core import (ROOT, atomic_json, atomic_torch, controller_for, digest,
                                      emit, load_model, read_json)
from steer_opd.rollout_topk import commit_topk_response


def run_job(job):
    if job["temperature"] != 1 or job["top_p"] != 1:
        raise ValueError("Controlled reference backend currently requires temperature=top_p=1")
    method = Method(**{**job["method"], "layers": tuple(job["method"]["layers"])})
    out = ROOT / job["output"]
    job, job_hash, pending = prepare_generation_manifest(out, job)
    if job.get("vector"):
        validate_request_vectors(ROOT / job["vector"], job["requests"])
    if not pending:
        atomic_json(out / "complete.json", {"requests": len(job["requests"]), "job_hash": job_hash,
                    "backend": "hf_cached_causal_gate_reference", "teacher_scores_saved": True})
        return
    os.environ["STEER_OPD_SUPPORT_SIZE"] = str(job["support_size"])
    student = load_model(ROOT / job["model"])
    teacher = load_model(ROOT / job["teacher"])
    tokenizer = AutoTokenizer.from_pretrained(ROOT / job["model"], local_files_only=True)
    stop_ids = {tokenizer.convert_tokens_to_ids(t) for t in ["<|im_end|>", "<|endoftext|>"]}
    stop_ids.discard(None)
    for request in pending:
        progress_path = out / "progress" / f"{request['key']}.pt"
        binding = digest({"job_hash": job_hash, "request": request})
        progress = None
        if progress_path.exists():
            saved = torch.load(progress_path, weights_only=True)
            if saved["binding"] != binding:
                raise ValueError("Controlled rollout progress provenance changed")
            progress = saved["state"]
        controller = controller_for(student, ROOT / job["vector"], request) if job.get("vector") else None
        try:
            row, scores = generate_response(student, teacher, request, controller, method,
                max_tokens=job["max_tokens"], support_size=job["support_size"], k=job["save_rollout_top_k"],
                stop_ids=stop_ids, progress=progress,
                save_progress=lambda state: atomic_torch(progress_path, {"binding": binding, "state": state}))
        finally:
            if controller:
                controller.detach()
        scores.update(teacher_model=job["teacher"], teacher_support_size=job["support_size"])
        row.update(text=tokenizer.decode(row["token_ids"], skip_special_tokens=True),
                   model=job["model"], steering=bool(job.get("vector")), job_hash=job_hash,
                   backend="hf_cached_causal_gate_reference", stop_reason=None)
        commit_topk_response(out, row, scores)
        if progress_path.exists():
            progress_path.unlink()
        emit("controlled_rollout_saved", key=request["key"], tokens=len(row["token_ids"]),
             injected=sum(row["steering_mask"]))
    atomic_json(out / "complete.json", {"requests": len(job["requests"]), "job_hash": job_hash,
                "backend": "hf_cached_causal_gate_reference", "teacher_scores_saved": True})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    run_job(read_json(args.job))


if __name__ == "__main__":
    main()
