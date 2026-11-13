#!/usr/bin/env python3
"""Real-model full-support and backward acceptance for optional method branches.

Never invoked by the zero-update diagnostic entry. Temporary updates affect
only this process; no experimental model checkpoint is saved.
"""

import argparse
import gc
import os
import torch

from steer_opd.controlled_rollout import generate_response
from steer_opd.direction_bank import bank_vectors, file_sha256
from steer_opd.method_config import resolve_method
from steer_opd.resumable_core import (ROOT, MasterAdam, atomic_json, controller_for, digest, load_model,
                                      project_logits, read_json, response_hidden, response_loss)
from steer_opd.rollout_topk import load_rollout_topk


def verify_full_replay(model, row, score, controller, last_only=False):
    with torch.inference_mode():
        hidden = response_hidden(model, row, controller)
        replay = project_logits(model, hidden[-1:] if last_only else hidden).log_softmax(-1)
        actual = score["full_rollout_logps"].to(replay.device)
        if replay.shape != actual.shape or not torch.isfinite(replay).all() or not torch.isfinite(actual).all():
            raise RuntimeError("Full-support branch probability shape/finite check failed")
        forward = (actual.exp() * (actual - replay)).sum(-1)
        reverse = (replay.exp() * (replay - actual)).sum(-1)
        passed = all(float(x.mean()) < 0.005 and float(x.max()) < 0.03 for x in (forward, reverse))
        entropy_error = None
        if "student_entropy" in score:
            entropy = -(actual.exp() * actual).sum(-1)
            entropy_error = float((entropy - score["student_entropy"].to(entropy.device)).abs().max())
            passed = passed and entropy_error < 1e-4
    if not passed:
        raise RuntimeError("Controlled branch full-support replay KL failed")
    return {"passed": passed, "support": actual.shape[-1], "positions": len(actual),
            "forward_mean_kl": float(forward.mean()), "forward_max_kl": float(forward.max()),
            "reverse_mean_kl": float(reverse.mean()), "reverse_max_kl": float(reverse.max()),
            "entropy_capture_max_error": entropy_error}


def accept(cfg, vector, rollouts=None):
    if cfg.get("stage") == "pre_opd_overlap_tuning" or cfg.get("optimizer_updates") == 0:
        raise ValueError("Backward acceptance is forbidden in the zero-update diagnostic")
    method = resolve_method(cfg)
    bank = torch.load(vector, weights_only=True) if vector else None
    os.environ["STEER_OPD_SUPPORT_SIZE"] = str(cfg["shared_vocabulary_size"])
    question = next(r for r in read_json(ROOT / cfg["data_dir"] / "opd.json") if bank is None or r["id"] in bank["valid_ids"])
    request = {"key": "branch_acceptance", "id": question["id"], "prompt_token_ids": question["student_prompt_ids"],
               "seed": cfg["seed"]}
    if bank:
        _, vector_id = bank_vectors(bank, question["id"])
        request.update(vector_id=vector_id, vector_hash=file_sha256(vector))
    source_model = ROOT / bank["identity"]["checkpoint"] if bank else ROOT / cfg["student"]
    student, teacher = load_model(source_model), load_model(ROOT / cfg["teacher"])
    controller = None
    distributions = []
    numeric_gate_override = False
    if rollouts is not None:
        manifest = read_json(rollouts / "job.json")["job"]
        numeric_gate_override = manifest.get("numeric_gate_override", False)
        for item in manifest["requests"]:
            row = read_json(rollouts / f"{item['key']}.json")
            score = load_rollout_topk(rollouts, row, cfg["supervision_top_k"], cfg["shared_vocabulary_size"])
            if "full_rollout_logps" not in score:
                paths = [rollouts / "full_logits" / f"{row['capture_request_id']}_{position}.pt"
                         for position in range(len(row["token_ids"]))]
                if not all(path.exists() for path in paths):
                    raise ValueError("Incomplete actual-backend distribution captures")
                score["full_rollout_logps"] = torch.stack([torch.load(path, weights_only=True).log_softmax(-1) for path in paths])
            controller = controller_for(student, vector, row)
            distributions.append(verify_full_replay(student, row, score, controller))
            if "full_teacher_logps" in score:
                checked = verify_full_replay(teacher, row, {"full_rollout_logps": score["full_teacher_logps"],
                    "student_entropy": score["teacher_entropy"]}, None)
                distributions.append({"component": "teacher", **checked})
            if "full_probe_logps" in score:
                for position in range(len(row["token_ids"])):
                    probe_row = {**row, "token_ids": row["token_ids"][:position+1],
                                 "steering_mask": row["steering_mask"][:position]+[False]}
                    checked = verify_full_replay(student, probe_row,
                        {"full_rollout_logps": score["full_probe_logps"][position:position+1]}, controller, last_only=True)
                    distributions.append({"component": "current_step_probe", "position": position, **checked})
            if controller:
                controller.detach()
        with torch.inference_mode():
            score["teacher_logps"] = project_logits(teacher, response_hidden(teacher, row)).log_softmax(-1).gather(
                -1, score["ids"].to(teacher.lm_head.weight.device).long()).cpu()
    else:
        controller = controller_for(student, vector, request)
        row, score = generate_response(student, teacher, request, controller, method, max_tokens=8,
                support_size=cfg["shared_vocabulary_size"], k=cfg["supervision_top_k"], stop_ids=set(), audit_full=True)
        distributions.append(verify_full_replay(student, row, score, controller))
        if controller:
            controller.detach()
    del student, teacher, controller
    gc.collect()
    torch.cuda.empty_cache()
    student = load_model(source_model, train=True)
    controller = controller_for(student, vector, row)
    optimizer = MasterAdam(student, cfg)
    loss, error = response_loss(student, row, score, controller, len(row["token_ids"]), cfg["logit_chunk_size"])
    optimizer.accumulate()
    norm = optimizer.step(cfg["grad_clip"])
    if not norm > 0:
        raise RuntimeError("Branch backward produced no gradient")
    if controller:
        controller.detach()
    atomic_json(ROOT / cfg["run_dir"] / ("branch_acceptance.json" if vector else "plain_batched_acceptance.json"), {
        "passed": True, "method_hash": digest(method.to_dict()), "bank_sha256": file_sha256(vector) if vector else None,
        "distributions": distributions, "loss": loss, "grad_norm": norm, "max_candidate_logp_error": error,
        "learning_replay_check": score["replay_validation"], "experimental_updates": 0,
        "numeric_gate_override": numeric_gate_override,
        "numeric_gate_purpose": "exercise both probe-reuse and formal-KV branches; experimental gate is CPU checked" if numeric_gate_override else None,
        "backend": cfg.get("generation_backend", "vllm_direction_bank" if method.gate == "all" else "hf_reference"),
        "vllm_per_question_bank_gpu_validated": cfg.get("generation_backend") != "hf_batched" and method.gate == "all" and method.granularity == "per_question"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--bank")
    parser.add_argument("--rollouts")
    args = parser.parse_args()
    accept(read_json(args.config), ROOT / args.bank if args.bank else None, ROOT / args.rollouts if args.rollouts else None)


if __name__ == "__main__":
    main()
