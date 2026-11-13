#!/usr/bin/env python3
"""HF scoring/extraction/training stages with persistent restart boundaries."""

import argparse
from contextlib import nullcontext
import gc
import json
import os
from pathlib import Path
import shutil
import time

import torch
from transformers import AutoTokenizer

from steer_opd.resumable_core import (
    ROOT, MasterAdam, atomic_json, atomic_torch, controller_for, digest, emit,
    load_model, project_logits, read_json, response_hidden, response_loss, score_hidden, set_seed,
)
from steer_opd.replay_validation import verify_short_replay


def model_path(cfg, group, step):
    if step == 0:
        return ROOT / cfg["student"]
    return ROOT / cfg["run_dir"] / group / "checkpoints" / f"step_{step:04d}" / "model"


def rollout_rows(path):
    job = read_json(Path(path) / "job.json")["job"]
    return [(r["key"], read_json(Path(path) / f"{r['key']}.json")) for r in job["requests"]]


def student_scores(cfg, group, step):
    run = ROOT / cfg["run_dir"]
    stage = run / group / "steps" / f"{step:04d}"
    rows = rollout_rows(stage / "rollouts")
    out = stage / "student_scores"
    if (out / "complete.json").exists():
        return
    model = load_model(model_path(cfg, group, step - 1))
    tokenizer = AutoTokenizer.from_pretrained(ROOT / cfg["student"], local_files_only=True)
    model._defined_vocab_size = len(tokenizer)
    controller = controller_for(model, run / "direction/vectors.pt" if group == "steer" else None)
    differences, invalid, validations = [], 0, []
    with torch.inference_mode():
        for index, (key, row) in enumerate(rows):
            target = out / f"{key}.pt"
            if target.exists():
                saved = torch.load(target, weights_only=True)
                if "short_replay_validation" in saved:
                    validations.append({"key": key, **saved["short_replay_validation"]})
                continue
            hidden = response_hidden(model, row, controller)
            score = score_hidden(model, hidden, cfg["supervision_top_k"], chunk=cfg["logit_chunk_size"],
                                 sampled=row["token_ids"])
            score["row_hash"] = digest(row)
            if row["sampled_logps"]:
                error = (score["sampled_logps"] - torch.tensor(row["sampled_logps"])).abs()
                score["rollout_replay_mean_logp_error"] = float(error.mean())
                score["rollout_replay_max_logp_error"] = float(error.max())
                differences.append(float(error.mean()))
                if float(error.mean()) > 0.12:
                    captures = stage / "rollouts" / "full_logits"
                    paths = [captures / f"{row.get('capture_request_id')}_{position}.pt"
                             for position in range(len(row["token_ids"]))]
                    if (len(paths) > 8 or not row.get("replay_audit_tokens")
                            or not all(path.exists() for path in paths)):
                        raise RuntimeError(f"Rollout/HF policy mismatch for {key}: mean logp error "
                                           f"{float(error.mean())}; full-distribution evidence unavailable")
                    captured = torch.stack([torch.load(path, weights_only=True) for path in paths]).cuda()
                    score["short_replay_validation"] = verify_short_replay(
                        captured, project_logits(model, hidden), row["token_ids"], row["sampled_logps"])
                    validations.append({"key": key, **score["short_replay_validation"]})
                    for position, path in enumerate(paths):
                        evidence = stage / "replay_evidence" / f"{key}_{position}.pt"
                        evidence.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(path, evidence)
                    emit("short_replay_verified", group=group, step=step, key=key,
                         sampled_mean_error=float(error.mean()), **score["short_replay_validation"])
            undefined_sampled = sum(t >= len(tokenizer) for t in row["token_ids"])
            invalid += undefined_sampled
            if undefined_sampled:
                raise ValueError("Student sampled an undefined tokenizer ID; cannot silently train it")
            if (score["ids"] >= len(tokenizer)).any():
                raise ValueError("Student Top-K includes undefined IDs; support protocol needs review")
            atomic_torch(target, score)
            emit("student_scored", group=group, step=step, row=index + 1, total=len(rows),
                 tokens=len(row["token_ids"]), undefined_mass=float(score["undefined_mass"].mean()))
    atomic_json(stage / "replay_validation.json", {"rows": len(rows),
                "short_full_distribution_checks": validations, "sampled_mean_error_limit": 0.12,
                "threshold_mean_kl": 0.005, "threshold_max_kl": 0.03})
    atomic_json(out / "complete.json", {"rows": len(rows), "undefined_sampled": invalid,
                                         "new_rows_mean_logp_errors": differences})


def teacher_scores(cfg, group, step):
    run = ROOT / cfg["run_dir"]
    stage = run / group / "steps" / f"{step:04d}"
    rows = rollout_rows(stage / "rollouts")
    out = stage / "scores"
    if (out / "complete.json").exists():
        return
    model = load_model(ROOT / cfg["teacher"])
    model._defined_vocab_size = len(AutoTokenizer.from_pretrained(ROOT / cfg["teacher"], local_files_only=True))
    with torch.inference_mode():
        for index, (key, row) in enumerate(rows):
            target = out / f"{key}.pt"
            if target.exists():
                continue
            student = torch.load(stage / "student_scores" / f"{key}.pt", weights_only=True)
            hidden = response_hidden(model, row)
            teacher = score_hidden(model, hidden, candidates=student["ids"], chunk=cfg["logit_chunk_size"])
            # Keep the full-softmax teacher log-probabilities on the exact fixed
            # student IDs, matching input_tokenizer=null in the reference fork.
            student["teacher_logps"] = teacher["logps"]
            student["teacher_undefined_mass"] = teacher["undefined_mass"]
            student["teacher_excluded_mass"] = teacher["excluded_mass"]
            atomic_torch(target, student)
            emit("teacher_scored", group=group, step=step, row=index + 1, total=len(rows))
    atomic_json(out / "complete.json", {"rows": len(rows), "teacher_input": cfg["input_protocol"]})


def extract(cfg):
    run = ROOT / cfg["run_dir"]
    out = run / "direction"
    if (out / "vectors.pt").exists():
        return
    texts = read_json(ROOT / cfg["data_dir"] / "direction.json")
    model = load_model(ROOT / cfg["student"])
    tokenizer = AutoTokenizer.from_pretrained(ROOT / cfg["student"], local_files_only=True)
    captured = []
    handle = model.model.layers[cfg["layer"]].register_forward_hook(
        lambda module, args, output: captured.append(output[0] if isinstance(output, tuple) else output))
    # Later causal decoder blocks cannot change an earlier block's activation.
    # Read only through the target block; no LM head or later blocks are needed.
    model.model.layers = torch.nn.ModuleList(list(model.model.layers[:cfg["layer"] + 1]))
    # Tokenizer offsets exclude role markers even when the Base tokenizer splits
    # the teacher's <think> strings into several ordinary tokens.
    import re
    excluded = set(tokenizer.all_special_ids)
    for index, item in enumerate(texts):
        target = out / "pooled" / f"{item['id']}.pt"
        if target.exists():
            continue
        records = [read_json(run / "calibration" / role / f"{item['id']}.json")
                   for role in ["teacher", "student"]]
        usable = all(r["finish_reason"] != "length" and r["text"].strip() for r in records)
        if not usable:
            atomic_torch(target, {"usable": False, "reason": "empty_or_truncated"})
            continue
        pools, norms, lengths = [], [], []
        for row in records:
            text = row["text"]
            # Detect severe exact repetition; keep correctness independent.
            words = text.split()
            if len(words) > 128 and len(set(words[-128:])) < 8:
                usable = False
                break
            encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            response = encoded["input_ids"]
            structural = [m.span() for m in re.finditer(r"</?(?:think|tool_response)>|<\|[^>]+\|>", text)]
            content_mask = [token not in excluded and b > a and
                            not any(a < end and b > begin for begin, end in structural)
                            for token, (a, b) in zip(response, encoded["offset_mapping"])]
            if not any(content_mask):
                usable = False
                break
            prefix = item["student_prompt_ids"]
            tokens = torch.tensor([prefix + response], device="cuda")
            captured.clear()
            with torch.inference_mode():
                model.model(input_ids=tokens, use_cache=False)
            states = captured[0][0, len(prefix):].float()
            states = states[torch.tensor(content_mask, device="cuda")]
            pools.append(states.mean(0).cpu())
            norms.append(states.norm(dim=-1).cpu())
            lengths.append(len(states))
            captured.clear()
        atomic_torch(target, {"usable": usable, "pools": pools if usable else [],
                              "norms": norms if usable else [], "lengths": lengths})
        emit("direction_pooled", question=index + 1, total=len(texts), usable=usable)
    handle.remove()
    valid = []
    for item in texts:
        payload = torch.load(out / "pooled" / f"{item['id']}.pt", weights_only=True)
        if payload["usable"]:
            valid.append((item["id"], payload))
    if len(valid) < 32:
        raise ValueError(f"Only {len(valid)} usable direction pairs")
    direction = torch.stack([v["pools"][0] - v["pools"][1] for _, v in valid]).mean(0)
    scale = torch.cat([n for _, v in valid for n in v["norms"]]).median()
    if not torch.isfinite(direction).all() or direction.norm() < 1e-8:
        raise ValueError("Degenerate direction")
    delta = cfg["rho"] * scale * direction / direction.norm()
    atomic_torch(out / "vectors.pt", {"direction": direction, "delta": delta,
                                       "layer": cfg["layer"], "scale": scale, "rho": cfg["rho"]})
    atomic_json(out / "metadata.json", {"usable": len(valid), "requested": len(texts),
                                         "ids": [i for i, _ in valid], "raw_norm": float(direction.norm()),
                                         "scale": float(scale), "delta_norm": float(delta.norm()),
                                         "layer": cfg["layer"], "rho": cfg["rho"], "direction": "T-S"})


def train(cfg, group, step):
    run = ROOT / cfg["run_dir"]
    stage = run / group / "steps" / f"{step:04d}"
    target = run / group / "checkpoints" / f"step_{step:04d}"
    if (target / "complete.json").exists():
        return
    pending = target.with_name(target.name + ".pending")
    if pending.exists():
        shutil.rmtree(pending)
    if shutil.disk_usage(ROOT).free < 28 * 1024**3:
        raise RuntimeError("Insufficient space for a safe atomic model/optimizer checkpoint; previous checkpoint retained")
    rows = rollout_rows(stage / "rollouts")
    assert (stage / "scores/complete.json").exists()
    set_seed(cfg["seed"])
    model = load_model(model_path(cfg, group, step - 1), train=True)
    optimizer = MasterAdam(model, cfg)
    if step > 1:
        state = torch.load(model_path(cfg, group, step - 1).parent / "resume.pt", map_location="cpu", weights_only=False)
        if state["config_hash"] != digest(cfg) or state["step"] != step - 1:
            raise ValueError("Resume configuration/step mismatch")
        optimizer.load(state["training"])
        del state
    controller = controller_for(model, run / "direction/vectors.pt" if group == "steer" else None)
    denominator = sum(len(row["token_ids"]) for _, row in rows)
    total_loss, max_error = 0.0, 0.0
    before = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for index, (key, row) in enumerate(rows):
        scores = torch.load(stage / "scores" / f"{key}.pt", weights_only=True)
        if scores["row_hash"] != digest(row):
            raise ValueError("Response changed after scoring")
        loss, error = response_loss(model, row, scores, controller, denominator, cfg["logit_chunk_size"])
        optimizer.accumulate()
        total_loss += loss
        max_error = max(max_error, error)
        if error > 0.2:
            raise RuntimeError(f"HF pre-update score/replay mismatch: {error}")
        emit("backward", group=group, step=step, row=index + 1, total=len(rows), loss=loss,
             replay_logp_max_error=error, elapsed=time.monotonic() - before)
    norm = optimizer.step(cfg["grad_clip"])
    # Directory visibility is the transaction boundary. Latest pointer changes
    # only after model AND optimizer/RNG checkpoint are durable.
    pending = target.with_name(target.name + ".pending")
    if pending.exists():
        shutil.rmtree(pending)  # Only this project's uncommitted transaction.
    pending.mkdir(parents=True)
    if controller:
        controller.detach()
    model.save_pretrained(pending / "model", safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / cfg["student"], local_files_only=True)
    tokenizer.save_pretrained(pending / "model")
    atomic_torch(pending / "resume.pt", {"step": step, "config_hash": digest(cfg),
                                         "training": optimizer.state_dict()})
    metrics = {"group": group, "step": step, "loss": total_loss, "grad_norm": norm,
               "tokens": denominator, "responses": len(rows), "seconds": time.monotonic() - before,
               "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
               "replay_max_logp_error": max_error, "config_hash": digest(cfg)}
    atomic_json(pending / "complete.json", metrics)
    os.replace(pending, target)
    atomic_json(run / group / "latest.json", {"step": step, "path": str(target.relative_to(ROOT)),
                                               "config_hash": digest(cfg)})
    atomic_json(stage / "train_metrics.json", metrics)
    # Keep latest resumable optimizer and evaluation model milestones. Never
    # remove shared inputs, user source files, or another project's artifacts.
    previous = target.with_name(f"step_{step - 1:04d}")
    if previous.exists():
        old_resume = previous / "resume.pt"
        if old_resume.exists():
            old_resume.unlink()
        if step - 1 not in cfg["eval_steps"]:
            shutil.rmtree(previous)
    for cache in [stage / "student_scores", stage / "scores", stage / "rollouts" / "full_logits"]:
        if cache.exists():
            shutil.rmtree(cache)
    emit("optimizer_committed", **metrics)


def diagnostic(cfg, group, step):
    run = ROOT / cfg["run_dir"]
    out = run / "evaluation" / ("base" if step == 0 else f"{group}_{step:04d}")
    if (out / "overlap.json").exists():
        return
    rows = rollout_rows(run / "fixed_prefixes")
    teacher_dir = run / "fixed_teacher_scores"
    if not (teacher_dir / "complete.json").exists():
        teacher = load_model(ROOT / cfg["teacher"])
        with torch.inference_mode():
            for key, row in rows:
                target = teacher_dir / f"{key}.pt"
                if target.exists():
                    continue
                score = score_hidden(teacher, response_hidden(teacher, row), chunk=cfg["logit_chunk_size"])
                atomic_torch(target, {"ids": score["ids"], "logps": score["logps"]})
        del teacher
        gc.collect()
        torch.cuda.empty_cache()
        atomic_json(teacher_dir / "complete.json", {"rows": len(rows)})
    student = load_model(model_path(cfg, group, step))
    total, intersection, student_mass, teacher_mass = 0, 0, 0.0, 0.0
    with torch.inference_mode():
        for key, row in rows:
            t = torch.load(teacher_dir / f"{key}.pt", weights_only=True)
            s = score_hidden(student, response_hidden(student, row), chunk=cfg["logit_chunk_size"])
            matches = s["ids"][:, :, None] == t["ids"][:, None, :]
            sm, tm = matches.any(-1), matches.any(-2)
            intersection += int(sm.sum())
            total += len(s["ids"])
            student_mass += float((s["logps"].exp() * sm).sum())
            teacher_mass += float((t["logps"].exp() * tm).sum())
    result = {"group": group, "step": step, "overlap16": intersection / (16 * total),
              "student_shared_mass": student_mass / total, "teacher_shared_mass": teacher_mass / total,
              "positions": total, "questions": len(rows), "student_hook": False,
              "prefixes": "fixed_initial_Base_dev_rollouts"}
    atomic_json(out / "overlap.json", result)
    emit("diagnostic_complete", **result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/figure8_first_run.json")
    parser.add_argument("--stage", choices=["student", "teacher", "extract", "train", "diagnostic"], required=True)
    parser.add_argument("--group", choices=["vanilla", "steer"])
    parser.add_argument("--step", type=int)
    args = parser.parse_args()
    cfg = read_json(args.config)
    os.environ["STEER_OPD_SUPPORT_SIZE"] = str(cfg["shared_vocabulary_size"])
    if args.stage == "extract":
        extract(cfg)
    else:
        {"student": student_scores, "teacher": teacher_scores, "train": train,
         "diagnostic": diagnostic}[args.stage](cfg, args.group, args.step)


if __name__ == "__main__":
    main()
