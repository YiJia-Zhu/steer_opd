#!/usr/bin/env python3
"""Read R once in S0; reuse the verified zero-update T/S pools and token cache."""

import argparse
import os

import torch
from transformers import AutoTokenizer

from steer_opd.direction_bank import file_sha256
from steer_opd.extraction import pool_question
from steer_opd.reference_controls import direction_geometry
from steer_opd.resumable_core import ROOT, atomic_json, atomic_torch, digest, load_model, read_json


def prepare(config):
    cfg = read_json(ROOT / config)
    run = ROOT / cfg["run_dir"]
    precheck = run / "precheck"
    output = run / "references/initial_pools"
    common = run / "references/common"
    bank = torch.load(precheck / "directions/initial/vectors.pt", weights_only=True)
    if bank["identity"]["checkpoint"] != cfg["student"] or bank["identity"]["step"] != 0:
        raise ValueError("Triplet cache must reuse the exact original S0")
    questions = read_json(common / "questions.json")
    if (output / "manifest.json").exists():
        saved = read_json(output / "manifest.json")
        if saved["checkpoint"] != cfg["student"]:
            raise ValueError("Initial pool extractor changed")
        return
    os.environ["STEER_OPD_SUPPORT_SIZE"] = str(cfg["shared_vocabulary_size"])
    model = load_model(ROOT / cfg["student"])
    layers = sorted(map(int, bank["layers"]))
    model.model.layers = torch.nn.ModuleList(list(model.model.layers[:max(layers)+1]))
    tokenizer = AutoTokenizer.from_pretrained(ROOT / cfg["student"], local_files_only=True)
    valid, rejected, hashes = [], dict(read_json(common / "manifest.json").get("initial_activation_filter", {})), {}
    for question in questions:
        qid = question["id"]
        target = output / f"{qid}.pt"
        records = {role: read_json(common / role / f"{qid}.json") for role in ("teacher", "reference", "student")}
        identity = {"checkpoint": cfg["student"], "step": 0, "question_hash": digest(question),
                    "text_hashes": {role: digest(row["text"]) for role, row in records.items()}, "layers": layers}
        if target.exists():
            value = torch.load(target, weights_only=True)
            if value["identity"] != identity:
                raise ValueError("Initial triplet provenance changed")
        else:
            previous = torch.load(precheck / "directions/initial/pooled" / f"{qid}.pt", weights_only=True)
            for role in ("teacher", "student"):
                original = read_json(precheck / "calibration" / role / f"{qid}.json")
                if original["text"] != records[role]["text"]:
                    raise ValueError("Cannot reuse T/S pooling after changing its text")
            artifacts = {}
            def save_reference(index, states):
                path = output / "token_activations" / f"{qid}_reference.pt"
                atomic_torch(path, {**states, "identity": identity})
                artifacts["reference"] = {"path": str(path.relative_to(ROOT)), "sha256": file_sha256(path)}
            ref = pool_question(model, tokenizer, question, [records["reference"]], layers, save_reference)
            value = {"identity": identity, "usable": previous["usable"] and ref["usable"], "roles": {},
                     "token_activations": artifacts,
                     "reused_TS_pool": str((precheck / "directions/initial/pooled" / f"{qid}.pt").relative_to(ROOT))}
            if value["usable"]:
                for layer in layers:
                    key = str(layer)
                    value["roles"][key] = {
                        "teacher": {"pool": previous["layers"][key]["pools"][0], "norm": previous["layers"][key]["norms"][0]},
                        "student": {"pool": previous["layers"][key]["pools"][1], "norm": previous["layers"][key]["norms"][1]},
                        "reference": {"pool": ref["layers"][key]["pools"][0], "norm": ref["layers"][key]["norms"][0]}}
                    geometry = direction_geometry({role: item["pool"] for role, item in value["roles"][key].items()})
                    if any(norm < 1e-8 for norm in geometry["raw_norms"].values()):
                        value["usable"] = False
                    value.setdefault("geometry", {})[key] = geometry
            atomic_torch(target, value)
        if value["usable"]:
            valid.append(question)
            hashes[qid] = file_sha256(target)
        else:
            rejected[qid] = "unusable_common_initial_triplet"
    if len(valid) < 8:
        raise ValueError("Fewer than8 common usable S0 triplets")
    # Freeze the same effective training pool for every current-pair branch.
    manifest = read_json(common / "manifest.json")
    manifest.update(valid_ids=[row["id"] for row in valid], initial_activation_filter=rejected)
    atomic_json(common / "questions.json", valid)
    atomic_json(common / "manifest.json", manifest)
    atomic_json(output / "manifest.json", {"checkpoint": cfg["student"], "step": 0, "layers": layers,
        "file_sha256": hashes, "valid_ids": manifest["valid_ids"], "rejected": rejected,
        "reference_token_activations": True, "teacher_student_token_activations": str((precheck / "directions/initial").relative_to(ROOT)),
        "parameter_updates": 0})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    prepare(parser.parse_args().config)
