#!/usr/bin/env python3
"""Versioned global/per-question extraction; no backward or optimizer."""

import argparse
from pathlib import Path
import os

import torch
from transformers import AutoTokenizer

from steer_opd.direction_bank import file_sha256, make_bank
from steer_opd.extraction import extract_pairs
from steer_opd.method_config import resolve_method
from steer_opd.resumable_core import ROOT, atomic_json, atomic_torch, digest, load_model, read_json


def extract_job(job):
    cfg = read_json(ROOT / job["config"])
    method = resolve_method({**cfg, "steering_method": job.get("method", cfg.get("steering_method", {}))})
    questions = read_json(ROOT / job["questions"])
    out = ROOT / job["output"]
    identity = {"checkpoint": job["extractor_model"], "step": job["extractor_step"],
                "questions_hash": digest(questions), "job_hash": digest(job),
                "reference_hashes": {role: {q["id"]: file_sha256(ROOT / directory / f"{q['id']}.json")
                                            for q in questions}
                                     for role, directory in job["role_directories"].items()}}
    target = out / "vectors.pt"
    if target.exists():
        bank = torch.load(target, weights_only=True)
        if bank["identity"] != identity:
            raise ValueError("Direction bank identity changed")
        return
    os.environ["STEER_OPD_SUPPORT_SIZE"] = str(cfg["shared_vocabulary_size"])
    model = load_model(ROOT / job["extractor_model"])
    tokenizer = AutoTokenizer.from_pretrained(ROOT / cfg["student"], local_files_only=True)
    if max(method.layers) >= len(model.model.layers):
        raise ValueError("Requested extraction layer does not exist")
    # Later blocks do not affect captured block outputs.
    model.model.layers = torch.nn.ModuleList(list(model.model.layers[:max(method.layers) + 1]))
    directories = {role: ROOT / path for role, path in job["role_directories"].items()}
    pooled = extract_pairs(model, tokenizer, questions, directories, method, out, identity)
    if sum(value["usable"] for value in pooled.values()) < job.get("minimum_pairs", 1):
        raise ValueError("Insufficient common usable calibration pairs")
    scales_path = ROOT / job["fixed_scales"]
    scales = torch.load(scales_path, weights_only=True) if scales_path.exists() else None
    if scales is None and job["extractor_step"] != 0:
        raise ValueError("Current-student refresh cannot invent a new activation scale")
    bank = make_bank(pooled, method, identity, fixed_scales=scales)
    if scales is None:
        atomic_torch(scales_path, {key: value["scale"] for key, value in bank["layers"].items()})
    atomic_torch(target, bank)
    atomic_json(out / "metadata.json", {"bank_id": bank["bank_id"], "identity": identity,
                "method": method.to_dict(), "valid_ids": bank["valid_ids"], "invalid": bank["invalid"],
                "candidate_count": len(questions), "valid_count": len(bank["valid_ids"]),
                "vector_file_sha256": file_sha256(target), "parameter_updates": 0})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    extract_job(read_json(args.job))


if __name__ == "__main__":
    main()
