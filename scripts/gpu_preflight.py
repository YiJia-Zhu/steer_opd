#!/usr/bin/env python3
"""Actual R1/JustRL forward/backward check, not a quality experiment."""

import json
import os
from pathlib import Path
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from steer_opd.math_data import grade_math, prompt_ids, read_rows
from steer_opd.steering import SteeringController, prediction_mask
from verl.trainer.distillation.fsdp.losses import tail_aware_kl_divergence


ROOT = Path(__file__).resolve().parents[1]
STUDENT = "/home/zhenning/data/huggingface_models/DeepSeek-R1-Distill-Qwen-1.5B"
TEACHER = "/home/zhenning/data/huggingface_datasets/JustRL-DeepSeek-1.5B"


def main():
    if not os.environ.get("STEER_OPD_GPU_ADMISSION"):
        raise RuntimeError("Launch through run_on_idle_gpus.py")
    started = time.monotonic()
    assert torch.cuda.device_count() == 1 and torch.cuda.is_available()
    print({"torch": torch.__version__, "device": torch.cuda.get_device_name(),
           "capability": torch.cuda.get_device_capability()}, flush=True)
    for answer, text, expected in [("2", r"Thus $\boxed{2}$.", True),
                                   ("2", r"Thus $\boxed{3}$.", False),
                                   (r"\frac{1}{2}", r"The answer is $\boxed{0.5}$.", True)]:
        assert grade_math(answer, text)["correct"] == expected
    tokenizer = AutoTokenizer.from_pretrained(STUDENT, local_files_only=True)
    row = read_rows(ROOT / "data/math500_pilot_v1/train_pilot.jsonl")[0]
    ids = torch.tensor([prompt_ids(tokenizer, row["question"])], device="cuda")
    model = AutoModelForCausalLM.from_pretrained(
        STUDENT, dtype=torch.float32, attn_implementation="sdpa", local_files_only=True,
    ).cuda().eval()
    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True,
    ).cuda().eval().requires_grad_(False)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        tokens = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=True,
                                temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=64,
                                pad_token_id=tokenizer.eos_token_id)
        expected = model(tokens, use_cache=False).logits.detach()
        teacher_logp = teacher(tokens, use_cache=False).logits.float().log_softmax(-1)
        teacher_top_logp, teacher_ids = teacher_logp.topk(16, -1)
    response = torch.zeros_like(tokens)
    response[:, ids.shape[1]:] = 1
    mask = prediction_mask(response)
    assert mask[0, ids.shape[1] - 1] and not mask[0, -1]
    direction = torch.linspace(-0.01, 0.01, model.config.hidden_size)
    controller = SteeringController(model.model.layers[14], direction, 0.0)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16), controller.loss_scope(mask):
        zero_alpha = model(tokens, use_cache=False).logits
    torch.testing.assert_close(zero_alpha, expected, rtol=0, atol=0)
    del expected, zero_alpha, teacher_logp
    controller.alpha = 1.0
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    before = controller.injected_forwards
    with controller.loss_scope(mask), torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(tokens, use_cache=False).logits
        student_logp = logits.float().log_softmax(-1).gather(-1, teacher_ids)
        token_loss, _, _, _ = tail_aware_kl_divergence(student_logp, teacher_top_logp, 1e-6)
        loss = token_loss[mask].mean()
        loss.backward()
    norm = torch.stack([p.grad.detach().float().norm() for p in model.parameters() if p.grad is not None]).norm()
    assert torch.isfinite(loss) and torch.isfinite(norm) and norm > 0
    assert all(p.grad is None for p in teacher.parameters())
    assert not controller.enabled and controller.mask is None
    controller.detach()
    report = {"scope": "Real-model numerical/gradient preflight; synthetic vector; no optimizer update or quality claim",
              "student": STUDENT, "teacher": TEACHER, "question_id": row["id"],
              "response_tokens": int(tokens.shape[1] - ids.shape[1]), "loss": loss.item(),
              "gradient_norm": norm.item(), "active_hook_calls": controller.injected_forwards - before,
              "zero_alpha_exact_match": True, "teacher_gradients_absent": True,
              "hook_removed": True, "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
              "elapsed_seconds": time.monotonic() - started, "gpu_admission": os.environ["STEER_OPD_GPU_ADMISSION"]}
    (ROOT / "runs/setup/gpu_preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
