#!/usr/bin/env python3
"""Replay teacher/student texts through S0, with equal weight per answer."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from steer_opd.math_data import read_rows
from steer_opd.runtime import verify_runtime


ROOT = Path(__file__).resolve().parents[1]
STUDENT = "/home/zhenning/data/huggingface_models/DeepSeek-R1-Distill-Qwen-1.5B"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-answers", type=Path, required=True)
    parser.add_argument("--student-answers", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--rho", type=float, default=0.03)
    args = parser.parse_args()
    verify_runtime()
    if not os.environ.get("STEER_OPD_GPU_ADMISSION"):
        raise RuntimeError("Launch through run_on_idle_gpus.py")
    out = args.output_dir.resolve()
    if not out.is_relative_to(ROOT) or (out / "vectors.pt").exists():
        raise ValueError("Use a new project-local output directory")
    out.mkdir(parents=True, exist_ok=True)
    teachers = {r["id"]: r for r in read_rows(args.teacher_answers)
                if r["correct"] and r["finish_reason"] != "length"}
    students = read_rows(args.student_answers)
    pairs = [(teachers[s["id"]], s) for s in students if s["id"] in teachers][:128]
    if len(pairs) < 32:
        raise ValueError(f"Only {len(pairs)} usable calibration pairs; need at least 32 for this pilot")
    tokenizer = AutoTokenizer.from_pretrained(STUDENT, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(STUDENT, local_files_only=True,
                                               dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    model.requires_grad_(False)
    captured = []
    handle = model.model.layers[args.layer].register_forward_hook(
        lambda module, inputs, output: captured.append(output[0] if isinstance(output, tuple) else output)
    )
    excluded = set(tokenizer.all_special_ids)
    for marker in ["<think>", "</think>"]:
        ids = tokenizer.encode(marker, add_special_tokens=False)
        if len(ids) == 1:
            excluded.add(ids[0])
    deltas, norms, lengths = [], [], []
    with torch.inference_mode():
        for index, (teacher, student) in enumerate(pairs):
            pools = []
            for record in [teacher, student]:
                # Shared R1 token mapping was checked; replay in S0's native prompt.
                prefix = student["prompt_token_ids"]
                response = record["token_ids"]
                tokens = torch.tensor([prefix + response], device="cuda")
                captured.clear()
                model.model(input_ids=tokens, use_cache=False)
                states = captured[0][0, len(prefix):].float()
                mask = torch.tensor([i not in excluded for i in response], device="cuda")
                if not mask.any():
                    raise ValueError("Calibration answer contains no content tokens")
                content = states[mask]
                pools.append(content.mean(0).cpu())
                norms.append(content.norm(dim=-1).cpu())
                lengths.append(int(mask.sum()))
            deltas.append(pools[0] - pools[1])
            print(json.dumps({"pair": index + 1, "total": len(pairs), "id": student["id"]}), flush=True)
    handle.remove()
    direction = torch.stack(deltas).mean(0)
    median_norm = torch.cat(norms).median().item()
    if not torch.isfinite(direction).all() or direction.norm().item() == 0:
        raise ValueError("Invalid steering direction")
    alpha = args.rho * median_norm / direction.norm().item()
    random_vector = torch.randn(direction.shape, generator=torch.Generator().manual_seed(42))
    random_vector *= direction.norm() / random_vector.norm()
    torch.save({"direction": direction, "random": random_vector}, out / "vectors.pt")
    metadata = {"student": STUDENT, "layer": args.layer, "rho": args.rho, "alpha": alpha,
                "pairs": len(pairs), "ids": [s["id"] for _, s in pairs],
                "median_activation_norm": median_norm, "direction_norm": direction.norm().item(),
                "teacher_content_lengths": lengths[::2], "student_content_lengths": lengths[1::2],
                "teacher_answers_sha256": hashlib.sha256(args.teacher_answers.read_bytes()).hexdigest(),
                "student_answers_sha256": hashlib.sha256(args.student_answers.read_bytes()).hexdigest()}
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    for group, key in [("E3", "direction"), ("E4", "random")]:
        settings = {"group": group, "vector_path": str(out / "vectors.pt"), "vector_key": key,
                    "layer": args.layer, "alpha": alpha, "auxiliary_updates": 20}
        (out / f"{group}.json").write_text(json.dumps(settings, indent=2) + "\n")


if __name__ == "__main__":
    main()
