#!/usr/bin/env python3
"""Read all reference roles in one frozen student; never update parameters."""

import argparse

import torch
from transformers import AutoTokenizer

from steer_opd.compass_banks import make_common_banks
from steer_opd.compass_job import project_path, model_path
from steer_opd.compass_alignment import ByteLevelCodec
from steer_opd.compass_scoring import load_text_scorer
from steer_opd.direction_bank import file_sha256
from steer_opd.extraction import pool_question
from steer_opd.execution_policy import require_execution_allowed
from steer_opd.reference_controls import usable_text
from steer_opd.resumable_core import atomic_json, atomic_torch, digest, read_json


def build(config):
    require_execution_allowed(config)
    questions = read_json(project_path(config["questions"]))
    roles = ("teacher", "reference", "student")
    output = project_path(config["bank_dir"])
    layers = config["layers"]
    tokenizer = AutoTokenizer.from_pretrained(model_path(config["student"]), local_files_only=True)
    model, pooled, identities = None, {}, {}
    for question in questions:
        records = [read_json(project_path(config["reference_dirs"][role]) / f"{question['id']}.json") for role in roles]
        if any(record["id"] != question["id"] or record.get("model") != config[role] for role, record in zip(roles, records)):
            raise ValueError("Reference text question/model differs from configured role")
        identity = {"checkpoint": config["student"], "step": 0, "layers": layers,
                    "question_hash": digest(question), "roles": list(roles),
                    "text_record_hashes": [digest(r) for r in records], "boundary": "block_output_before_final_norm"}
        identities[question["id"]] = identity
        target = output / "pooled" / f"{question['id']}.pt"
        if target.exists():
            value = torch.load(target, weights_only=True, map_location="cpu")
            if value["identity"] != identity:
                raise ValueError("Cached extraction changed checkpoint/texts/question/layer")
            for artifact in value.get("token_activations", {}).values():
                if file_sha256(project_path(artifact["path"])) != artifact["sha256"]:
                    raise ValueError("Saved role activation artifact changed")
        elif not all(usable_text(record) for record in records):
            value = {"usable": False, "reason": "common_text_quality_filter", "identity": identity}
            atomic_torch(target, value)
        else:
            if model is None:
                model = load_text_scorer(model_path(config["student"]), ByteLevelCodec(tokenizer))
                # Outputs up to the extraction point are identical; later
                # blocks cannot affect an earlier block's captured activation.
                model.model.layers = torch.nn.ModuleList(list(model.model.layers[:max(layers)+1]))
            artifacts = {}
            def save(index, payload):
                p = output / "token_activations" / question["id"] / f"{roles[index]}.pt"
                atomic_torch(p, {**payload, "identity": identity, "role": roles[index]})
                artifacts[roles[index]] = {"path": str(p.relative_to(project_path("."))), "sha256": file_sha256(p)}
            value = {**pool_question(model, tokenizer, question, records, layers, save,
                                   cache_dtype=torch.bfloat16, allow_zero=True),
                     "identity": identity, "token_activations": artifacts}
            atomic_torch(target, value)
        pooled[question["id"]] = value
    valid = [q for q in questions if pooled[q["id"]]["usable"]]
    if len(valid) < config.get("minimum_pairs", 8):
        raise ValueError("Too few common valid reference triplets")
    banks, scales = make_common_banks(pooled, layers,
        {"checkpoint": config["student"], "step": 0, "reference_pool_hash": digest(identities)}, alpha=config["alpha"])
    hashes = {}
    for name, bank in banks.items():
        p = output / f"{name}.pt"
        atomic_torch(p, bank)
        hashes[name] = file_sha256(p)
    atomic_torch(output / "common_scales.pt", scales)
    atomic_json(output / "questions.json", valid)
    atomic_json(output / "manifest.json", {"valid_ids": [q["id"] for q in valid], "bank_sha256": hashes,
        "rejected": {qid: v.get("reason", "unusable_activation") for qid, v in pooled.items() if not v["usable"]},
        "identity_hash": digest(identities), "parameter_updates": 0, "layers": layers, "alpha": config["alpha"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    build(read_json(args.config))


if __name__ == "__main__":
    main()
