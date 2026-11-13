#!/usr/bin/env python3
"""Versioned global/per-question extraction; no backward or optimizer."""

import argparse
from pathlib import Path
import os

import torch
from transformers import AutoTokenizer

from steer_opd.direction_bank import file_sha256, make_bank
from steer_opd.extraction import extract_pairs, validate_token_artifacts
from steer_opd.method_config import resolve_method
from steer_opd.resumable_core import ROOT, atomic_json, atomic_torch, digest, load_model, read_json


def write_metadata(out, bank, pooled, job):
    atomic_json(out / "metadata.json", {"bank_id": bank["bank_id"], "identity": bank["identity"],
                "method": bank["method"], "valid_ids": bank["valid_ids"], "invalid": bank["invalid"],
                "candidate_count": len(pooled), "valid_count": len(bank["valid_ids"]),
                "token_activation_cache": job.get("cache_token_activations", False),
                "token_activation_dtype": job.get("token_activation_dtype", "bfloat16"),
                "shared_initial_pool": job.get("shared_initial_pool"),
                "token_activation_artifacts": {qid: item.get("token_activations", {}) for qid, item in pooled.items()},
                "vector_file_sha256": file_sha256(out / "vectors.pt"), "parameter_updates": 0})


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
        pooled = {q["id"]: torch.load(out / "pooled" / f"{q['id']}.pt", weights_only=True) for q in questions}
        if job.get("cache_token_activations"):
            for value in pooled.values():
                validate_token_artifacts(value, out, method.roles)
        # Repair a kill between tensor commit and the small metadata commit
        # without loading a model or repeating the two reference reads.
        write_metadata(out, bank, pooled, job)
        return
    if job.get("shared_initial_pool") and job["extractor_step"] == 0:
        source = ROOT / job["shared_initial_pool"]
        manifest = read_json(source / "manifest.json")
        if manifest["checkpoint"] != job["extractor_model"] or manifest["step"] != 0:
            raise ValueError("Shared initial pools use another extractor")
        pooled = {}
        for question in questions:
            qid = question["id"]
            path = source / f"{qid}.pt"
            if file_sha256(path) != manifest["file_sha256"][qid]:
                raise ValueError("Shared initial pool changed")
            item = torch.load(path, weights_only=True)
            if digest(question) != item["identity"]["question_hash"]:
                raise ValueError("Shared initial question changed")
            for role in method.roles:
                text = read_json(ROOT / job["role_directories"][role] / f"{qid}.json")["text"]
                if digest(text) != item["identity"]["text_hashes"][role]:
                    raise ValueError("Cannot reuse activation pools for modified texts")
            layers = {str(layer): {"pools": [item["roles"][str(layer)][role]["pool"] for role in method.roles],
                      "norms": [item["roles"][str(layer)][role]["norm"] for role in method.roles]} for layer in method.layers}
            pooled[qid] = {"usable": item["usable"], "layers": layers, "pair_hash": digest({"identity": identity, "qid": qid}),
                           "shared_initial_pool_file": str(path.relative_to(ROOT))}
            atomic_torch(out / "pooled" / f"{qid}.pt", pooled[qid])
    else:
        os.environ["STEER_OPD_SUPPORT_SIZE"] = str(cfg["shared_vocabulary_size"])
        model = load_model(ROOT / job["extractor_model"])
        tokenizer = AutoTokenizer.from_pretrained(ROOT / cfg["student"], local_files_only=True)
        if max(method.layers) >= len(model.model.layers):
            raise ValueError("Requested extraction layer does not exist")
        # Later blocks do not affect captured block outputs.
        model.model.layers = torch.nn.ModuleList(list(model.model.layers[:max(method.layers) + 1]))
        directories = {role: ROOT / path for role, path in job["role_directories"].items()}
        pooled = extract_pairs(model, tokenizer, questions, directories, method, out, identity,
                               cache_token_activations=job.get("cache_token_activations", False),
                               cache_dtype=job.get("token_activation_dtype", "bfloat16"))
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
    write_metadata(out, bank, pooled, job)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    extract_job(read_json(args.job))


if __name__ == "__main__":
    main()
