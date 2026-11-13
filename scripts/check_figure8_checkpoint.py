#!/usr/bin/env python3
"""CPU reload verification of a committed real-model checkpoint."""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from steer_opd.resumable_core import atomic_json, read_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    root = args.checkpoint.resolve()
    committed = read_json(root / "complete.json")
    state = torch.load(root / "resume.pt", map_location="cpu", mmap=True, weights_only=False)
    assert state["step"] == committed["step"]
    assert state["config_hash"] == committed["config_hash"]
    model = AutoModelForCausalLM.from_pretrained(root / "model", local_files_only=True,
                                               dtype=torch.bfloat16, device_map="cpu")
    named = list(model.named_parameters())
    training = state["training"]
    assert [n for n, _ in named] == training["names"]
    for (name, value), master in zip(named, training["master"]):
        assert master.dtype == torch.float32
        assert torch.equal(value, master.to(value.dtype)), name
    optimizer = training["optimizer"]
    steps = [int(s["step"]) for s in optimizer["state"].values()]
    assert steps and min(steps) == max(steps) == state["step"]
    assert training["torch_rng"].dtype == torch.uint8
    assert training["cuda_rng"] and training["cuda_rng"][0].dtype == torch.uint8
    result = {"passed": True, "step": state["step"], "parameter_tensors": len(named),
              "saved_model_equals_bf16_master": True, "adam_steps": sorted(set(steps)),
              "cpu_cuda_python_numpy_rng_present": all(k in training for k in
                  ["torch_rng", "cuda_rng", "python_rng", "numpy_rng"]), "gpu_used": False}
    atomic_json(root / "reload_verification.json", result)
    print(result)


if __name__ == "__main__":
    main()
