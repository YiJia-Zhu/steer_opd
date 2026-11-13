#!/usr/bin/env python3
"""Memory-admitted batched causal rollout, per-token resumable and scored once."""

import argparse
import gc
import os
import time

import torch
from transformers import AutoTokenizer

from steer_opd.batched_controlled import (BatchedController, batched_step, forward_rows, BatchMemoryPressure,
                                          available_memory, require_memory)
from steer_opd.direction_bank import bank_vectors, validate_request_vectors
from steer_opd.generation_runtime import prepare_generation_manifest
from steer_opd.method_config import Method
from steer_opd.resumable_core import ROOT, atomic_json, atomic_torch, digest, emit, load_model, read_json
from steer_opd.rollout_topk import HF_SOURCE, commit_topk_response


TENSOR_FIELDS = ("ids", "logps", "teacher_logps", "teacher_topk_ids", "teacher_topk_logps",
                 "probe_topk_ids", "probe_topk_logps")
ENTROPY_FIELDS = ("student_entropy", "teacher_entropy", "probe_entropy")


class RepackBatch(RuntimeError):
    pass


def empty_state():
    return {key: [] for key in (*TENSOR_FIELDS, *ENTROPY_FIELDS, "token_ids", "steering_mask",
                                "gate_metrics", "sampled_logps", "teacher_sampled_logps")} | {"gate_history": {}}


def capacity(student, teacher, job, pending):
    def kv_bytes(model):
        cfg = model.config
        return 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * cfg.head_dim * 2
    free = available_memory()
    # Admit a larger initial batch; monitor temporary KV allocations at every
    # step and split/repack from durable32-token checkpoints if context grows.
    longest = max(len(row["prompt_token_ids"]) for row in pending) + min(1024, job["max_tokens"])
    # Resident + temporary append-only probe/formal cache tensors and headroom.
    per_request = longest * (4 * kv_bytes(student) + 2 * kv_bytes(teacher))
    available = max(0, int(free * 0.8) - 2 * 1024**3)
    count = min(len(pending), available // per_request)
    if count < 1:
        raise RuntimeError("Insufficient free memory for one exact gated rollout at the frozen token budget")
    return int(count)


def run_batch(student, teacher, tokenizer, requests, job, job_hash, output, bank):
    method = Method(**{**job["method"], "layers": tuple(job["method"]["layers"])})
    if method.operator != "additive" or method.gate == "static_overlap":
        raise ValueError("Use run_static_overlap_screen.py for Slerp or frozen static masks")
    device = student.lm_head.weight.device
    states, samplers, gaters, bindings = [], [], [], []
    for request in requests:
        binding = digest({"job_hash": job_hash, "request": request})
        path = output / "progress" / f"{request['key']}.pt"
        saved = torch.load(path, weights_only=True) if path.exists() else None
        if saved and saved["binding"] != binding:
            raise ValueError("Batched progress provenance changed")
        state = saved["state"] if saved else empty_state()
        if job.get("capture_logits"):
            state.setdefault("full_rollout_logps", [])
            state.setdefault("full_teacher_logps", [])
            state.setdefault("full_probe_logps", [])
        sample = torch.Generator(device=device).manual_seed(request["seed"])
        gate = torch.Generator().manual_seed(request["seed"] + 87654321)
        if saved:
            sample.set_state(state["sampling_rng"])
            gate.set_state(state["gating_rng"])
        states.append(state)
        samplers.append(sample)
        gaters.append(gate)
        bindings.append(binding)
    controller = None
    if bank:
        all_vectors = [bank_vectors(bank, request["id"])[0] for request in requests]
        controller = BatchedController(student, {layer: torch.stack([value[layer] for value in all_vectors]).to(device)
                                                for layer in all_vectors[0]})
    stop = {tokenizer.convert_tokens_to_ids(value) for value in ["<|im_end|>", "<|endoftext|>"]} - {None}
    if job.get("ignore_eos"):
        if not job.get("capture_logits"):
            raise ValueError("Ignore-EOS requires a numerical acceptance capture")
        stop = set()
    completed = [False] * len(requests)
    def save(index):
        state = states[index]
        state["sampling_rng"], state["gating_rng"] = samplers[index].get_state(), gaters[index].get_state()
        atomic_torch(output / "progress" / f"{requests[index]['key']}.pt",
                     {"binding": bindings[index], "state": state})
    def finish(index):
        request, state = requests[index], states[index]
        scores = {key: torch.stack(state[key]) for key in TENSOR_FIELDS if state[key]}
        scores.update({key: torch.tensor(state[key], dtype=torch.float32) for key in ENTROPY_FIELDS if state[key]})
        if job.get("capture_logits"):
            scores["full_rollout_logps"] = torch.stack(state["full_rollout_logps"])
            scores["full_teacher_logps"] = torch.stack(state["full_teacher_logps"])
            if state["full_probe_logps"]:
                scores["full_probe_logps"] = torch.stack(state["full_probe_logps"])
        scores.update(source=HF_SOURCE, probability_mode="raw_logprobs", k=job["save_rollout_top_k"],
            support_size=job["support_size"], entropy_support_size=job["support_size"],
            sampled_logps=torch.tensor(state["sampled_logps"], dtype=torch.float32),
            teacher_model=job["teacher"], teacher_support_size=job["support_size"])
        scores["teacher_sampled_logps"] = torch.tensor(state["teacher_sampled_logps"], dtype=torch.float32)
        row = {**request, **{key: state[key] for key in ["token_ids", "steering_mask", "gate_metrics", "sampled_logps"]},
            "text": tokenizer.decode(state["token_ids"], skip_special_tokens=True), "model": job["model"],
            "steering": bool(bank), "job_hash": job_hash, "backend": "hf_batched_causal_gate",
            "finish_reason": "stop" if state["token_ids"][-1] in stop else "length", "stop_reason": None,
            "token_record_schema": 1, "prediction_start": len(request["prompt_token_ids"]) - 1}
        commit_topk_response(output, row, scores)
        path = output / "progress" / f"{request['key']}.pt"
        if path.exists():
            path.unlink()
        completed[index] = True
        emit("batched_rollout_saved", key=request["key"], tokens=len(row["token_ids"]),
             injected=sum(row["steering_mask"]))
    try:
        with torch.inference_mode():
            histories = [request["prompt_token_ids"] + state["token_ids"] for request, state in zip(requests, states)]
            width = max(len(tokens) - 1 for tokens in histories)
            if min(map(len, histories)) < 2:
                raise ValueError("Expected a nonempty cached prompt history")
            def kv_bytes(model):
                value = model.config
                return 2*value.num_hidden_layers*value.num_key_value_heads*value.head_dim*2
            require_memory(len(requests)*width*(kv_bytes(student)+kv_bytes(teacher)) + 4*len(requests)*width*width)
            pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            inputs = torch.full((len(requests), width), pad, dtype=torch.long, device=device)
            attention = torch.zeros_like(inputs, dtype=torch.bool)
            past_mask = torch.zeros_like(attention)
            for index, (tokens, request, state) in enumerate(zip(histories, requests, states)):
                offset = width - len(tokens) + 1
                inputs[index, offset:] = torch.tensor(tokens[:-1], device=device)
                attention[index, offset:] = True
                start = offset + len(request["prompt_token_ids"]) - 1
                past_mask[index, start:] = torch.tensor(state["steering_mask"], device=device)
            # Reconstruct exactly recorded historical intervention once on resume.
            _, student_cache = forward_rows(student, inputs, None, attention, controller, past_mask)
            _, teacher_cache = forward_rows(teacher, inputs, None, attention)
            tokens = torch.tensor([[value[-1]] for value in histories], device=device)
            for index, state in enumerate(states):
                if state["token_ids"] and (state["token_ids"][-1] in stop or len(state["token_ids"]) >= job["max_tokens"]):
                    finish(index)
            while not all(completed):
                active = [not value for value in completed]
                attention = torch.cat([attention, torch.tensor(active, device=device)[:, None]], dim=1)
                draws = [float(torch.rand((), generator=gate)) if live else 0.0 for gate, live in zip(gaters, active)]
                p, q, student_cache, teacher_cache, injected, metrics, probe = batched_step(student, teacher, tokens,
                    student_cache, teacher_cache, attention, controller, method,
                    [len(state["token_ids"]) for state in states], draws,
                    [state["gate_history"] for state in states], active,
                    record_probe=job.get("record_comparison_probe", False))
                if p.shape[1] != job["support_size"] or q.shape != p.shape or not torch.isfinite(p).all() or not torch.isfinite(q).all():
                    raise ValueError("Invalid batched policy distribution/support")
                pv, pi = p.topk(job["save_rollout_top_k"])
                qv, qi = q.topk(job["save_rollout_top_k"])
                ent_p = -(p.exp() * p).sum(-1)
                ent_q = -(q.exp() * q).sum(-1)
                buffers = {"ids": pi.int().cpu(), "logps": pv.float().cpu(),
                    "teacher_logps": q.gather(-1, pi).float().cpu(), "teacher_topk_ids": qi.int().cpu(),
                    "teacher_topk_logps": qv.float().cpu()}
                if probe is not None:
                    value, index = probe.topk(job["save_rollout_top_k"])
                    buffers.update(probe_topk_ids=index.int().cpu(), probe_topk_logps=value.float().cpu())
                    ent_probe = -(probe.exp() * probe).sum(-1)
                for index, live in enumerate(active):
                    if not live:
                        continue
                    state = states[index]
                    token = int(torch.multinomial(p[index].exp(), 1, generator=samplers[index]))
                    tokens[index, 0] = token
                    state["token_ids"].append(token)
                    state["steering_mask"].append(injected[index])
                    state["gate_metrics"].append(metrics[index])
                    state["sampled_logps"].append(float(p[index, token]))
                    state["teacher_sampled_logps"].append(float(q[index, token]))
                    if job.get("capture_logits"):
                        state["full_rollout_logps"].append(p[index].cpu())
                        state["full_teacher_logps"].append(q[index].cpu())
                        if probe is not None:
                            state["full_probe_logps"].append(probe[index].cpu())
                    state["student_entropy"].append(float(ent_p[index]))
                    state["teacher_entropy"].append(float(ent_q[index]))
                    if probe is not None:
                        state["probe_entropy"].append(float(ent_probe[index]))
                    for key, values in buffers.items():
                        state[key].append(values[index].clone())
                    if token in stop or len(state["token_ids"]) >= job["max_tokens"]:
                        finish(index)
                    elif len(state["token_ids"]) % 32 == 0:
                        save(index)
                if (len(requests) >= 4 and not all(completed) and sum(completed)*2 >= len(requests)
                        and all(completed[index] or len(state["token_ids"]) % 32 == 0 for index, state in enumerate(states))):
                    # Every unfinished request is already durable. Refill the
                    # batch instead of computing padding for many ended rows.
                    raise RepackBatch("Refill after half the batch has completed")
    finally:
        if controller:
            controller.detach()


def run_job(job):
    if job["temperature"] != 1 or job["top_p"] != 1:
        raise ValueError("Batched backend requires temperature=top_p=1")
    output = ROOT / job["output"]
    job, job_hash, pending = prepare_generation_manifest(output, job)
    if not pending:
        atomic_json(output / "complete.json", {"job_hash": job_hash, "requests": len(job["requests"])})
        return
    os.environ["STEER_OPD_SUPPORT_SIZE"] = str(job["support_size"])
    bank = None
    if job.get("vector"):
        validate_request_vectors(ROOT / job["vector"], pending)
        bank = torch.load(ROOT / job["vector"], weights_only=True)
    student, teacher = load_model(ROOT / job["model"]), load_model(ROOT / job["teacher"])
    tokenizer = AutoTokenizer.from_pretrained(ROOT / job["model"], local_files_only=True)
    batch_size = capacity(student, teacher, job, pending)
    atomic_json(output / "runtime.json", {"backend": "hf_batched_causal_gate", "memory_admitted_batch": batch_size,
        "requests": len(pending), "shared_teacher_scoring": True, "gpu_count": 1})
    started = time.monotonic()
    queue = list(pending)
    events = []
    while queue:
        memory_pressure, repack, succeeded = False, False, False
        try:
            run_batch(student, teacher, tokenizer, queue[:batch_size], job, job_hash, output, bank)
            succeeded = True
        except RepackBatch:
            repack = True
        except (BatchMemoryPressure, torch.OutOfMemoryError) as exc:
            if batch_size == 1:
                raise RuntimeError("One request cannot fit the frozen context budget; artifacts retained") from exc
            memory_pressure = True
            emit("batched_memory_split", previous_batch=batch_size, reason=type(exc).__name__,
                 maximum_recomputed_tokens_per_unfinished_request=31)
        # Clear exception tracebacks before freeing tensors and retrying.
        gc.collect()
        torch.cuda.empty_cache()
        queue = [request for request in queue if not (output/f"{request['key']}.json").exists()]
        events.append({"batch": batch_size, "memory_split": memory_pressure, "repack": repack,
                       "completed_batch": succeeded, "remaining": len(queue)})
        if queue:
            batch_size = max(1, batch_size//2) if memory_pressure else capacity(student, teacher, job, queue)
        atomic_json(output/"batch_schedule.json", {"events": events, "scheduling_changes_do_not_change_gate_definition": True})
    atomic_json(output / "complete.json", {"job_hash": job_hash, "requests": len(job["requests"]),
        "seconds": time.monotonic() - started, "teacher_scores_saved": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    run_job(read_json(parser.parse_args().job))
