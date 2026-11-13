#!/usr/bin/env python3
"""Resumable vLLM generation: one atomically committed record per request."""

import argparse
import json
import os
from pathlib import Path
import time

from steer_opd.resumable_core import ROOT, atomic_json, emit, read_json, require_gpu
from steer_opd.generation_runtime import prepare_generation_manifest
from steer_opd.rollout_topk import commit_topk_response, extract_topk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    job = read_json(args.job)
    out = ROOT / job["output"]
    out.mkdir(parents=True, exist_ok=True)
    job, job_hash, pending = prepare_generation_manifest(out, job)
    if not pending:
        atomic_json(out / "complete.json", {"requests": len(job["requests"]), "job_hash": job_hash})
        return
    if job.get("method"):
        from steer_opd.direction_bank import validate_request_vectors
        validate_request_vectors(ROOT / job["vector"], pending)
    require_gpu()
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    # Installed nvcc 13.2 and runtime headers 13.0 cannot build FlashInfer's
    # optional sampler. Use vLLM's seeded native sampler and prebuilt FA2;
    # no shared environment packages need modification.
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    os.environ["OMP_NUM_THREADS"] = "1"
    from vllm import LLM, SamplingParams

    max_len = job["max_tokens"] + max(len(r["prompt_token_ids"]) for r in job["requests"])
    use_graph = job.get("decode_cuda_graph", False)
    save_topk = job.get("save_rollout_top_k")
    if save_topk and (job.get("logprobs") != save_topk or job.get("logprobs_mode") != "raw_logprobs"):
        raise ValueError("Saved rollout Top-K requires matching raw_logprobs")
    extra = {"worker_cls": "steer_opd.vllm_worker.SteeringWorker"}
    if save_topk:
        extra["logprobs_mode"] = "raw_logprobs"
    os.environ["STEER_OPD_VLLM_SUPPORT"] = str(job["support_size"])
    os.environ["STEER_OPD_REFERENCE_OPS"] = "1" if job.get("normalization") == "hf_bf16_reference" else "0"
    audit_tokens = job.get("replay_audit_tokens", 0)
    os.environ["STEER_OPD_REPLAY_AUDIT_TOKENS"] = str(audit_tokens)
    os.environ["STEER_OPD_CAPTURE_ALL"] = "1" if save_topk and job.get("capture_logits") else "0"
    if job.get("capture_logits") or audit_tokens:
        os.environ["STEER_OPD_CAPTURE_DIR"] = str(out / "full_logits")
    else:
        os.environ.pop("STEER_OPD_CAPTURE_DIR", None)
    if job.get("vector"):
        os.environ["STEER_OPD_VLLM_VECTOR"] = str(ROOT / job["vector"])
    else:
        os.environ.pop("STEER_OPD_VLLM_VECTOR", None)
    if use_graph:
        extra["compilation_config"] = {
            "mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY",
        }
    llm = LLM(
        model=str(ROOT / job["model"]), dtype="bfloat16", tensor_parallel_size=1,
        gpu_memory_utilization=job.get("gpu_memory_utilization", 0.75),
        max_model_len=max_len, enforce_eager=not use_graph,
        enable_prefix_caching=False, enable_chunked_prefill=False,
        generation_config="vllm", seed=job.get("seed", 42),
        hf_overrides={"head_dtype": job.get("head_dtype", "float32")},
        max_logprobs=20, disable_log_stats=False, **extra,
    )
    engine_cfg = llm.llm_engine.vllm_config
    if save_topk and engine_cfg.model_config.logprobs_mode != "raw_logprobs":
        raise ValueError("vLLM did not resolve the required raw probability mode")
    resolved = {"job_hash": job_hash, "pending_requests": len(pending),
                "total_requests": len(job["requests"]),
                "scheduler": "vllm_defaults",
                "resolved_max_num_seqs": engine_cfg.scheduler_config.max_num_seqs,
                "resolved_max_num_batched_tokens": engine_cfg.scheduler_config.max_num_batched_tokens,
                "resolved_graph_max_size": engine_cfg.compilation_config.max_cudagraph_capture_size,
                "resolved_graph_sizes": engine_cfg.compilation_config.cudagraph_capture_sizes,
                "resolved_logprobs_mode": engine_cfg.model_config.logprobs_mode}
    atomic_json(out / "runtime.json", resolved)
    emit("resolved_generation_runtime", **resolved)
    if job.get("vector"):
        emit("vllm_hook_installed", result=llm.collective_rpc("project_steering_status"))
    tokenizer = llm.get_tokenizer()
    stop_ids = [tokenizer.convert_tokens_to_ids(t) for t in ["<|im_end|>", "<|endoftext|>"]]
    stop_ids = sorted({i for i in stop_ids if i is not None})
    from vllm.sampling_params import RequestOutputKind
    params = [SamplingParams(
        n=1, temperature=job["temperature"], top_p=job["top_p"], top_k=-1,
        repetition_penalty=1.0, max_tokens=job["max_tokens"], stop_token_ids=stop_ids,
        seed=r["seed"], logprobs=job.get("logprobs", 0), output_kind=RequestOutputKind.FINAL_ONLY,
        min_tokens=job.get("benchmark_min_tokens", 0),
    ) for r in pending]
    request_ids = llm.enqueue([{"prompt_token_ids": r["prompt_token_ids"]} for r in pending],
                              params, use_tqdm=False)
    # vLLM 0.32 enqueue returns internal "counter-UUID" IDs, whereas finished
    # RequestOutput uses the external counter. Preserve their exact association.
    by_id = {key.rsplit("-", 1)[0]: row for key, row in zip(request_ids, pending)}
    internal_ids = {key.rsplit("-", 1)[0]: key for key in request_ids}
    if job.get("per_question_vectors"):
        llm.collective_rpc("project_bind_vectors", args=({key: row["id"] for key, row in by_id.items()},))
    before = time.monotonic()
    completed = len(job["requests"]) - len(pending)
    generated_tokens = 0
    while llm.llm_engine.has_unfinished_requests():
        for generated in llm.llm_engine.step():
            if not generated.finished:
                continue
            request = by_id.pop(generated.request_id)
            sample = generated.outputs[0]
            logps = []
            if sample.logprobs is not None:
                logps = [float(step[token].logprob) for token, step in zip(sample.token_ids, sample.logprobs)]
            row = {**request, "token_ids": list(sample.token_ids), "text": sample.text,
                   "finish_reason": sample.finish_reason, "stop_reason": sample.stop_reason,
                   "sampled_logps": logps, "model": job["model"], "steering": bool(job.get("vector")),
                   "job_hash": job_hash, "vllm_request_id": generated.request_id,
                   "capture_request_id": internal_ids[generated.request_id]}
            if (job.get("logprobs") or 0) > 0 and not save_topk:
                row["top_logprobs"] = [{str(k): float(v.logprob) for k, v in step.items()}
                                       for step in sample.logprobs]
            if job.get("capture_logits") or audit_tokens:
                row["capture_request_id"] = llm.collective_rpc("project_capture_ids")[0].get(generated.request_id)
            if audit_tokens:
                row["replay_audit_tokens"] = audit_tokens
                # Long replies use the normal sampled-logprob check. Retain
                # complete original distributions only for short replies.
                if len(sample.token_ids) > audit_tokens:
                    for path in (out / "full_logits").glob(f"{row['capture_request_id']}_*.pt"):
                        path.unlink()
            if save_topk:
                score = extract_topk(row["token_ids"], sample.logprobs, save_topk, job["support_size"])
                commit_topk_response(out, row, score)
            else:
                atomic_json(out / f"{request['key']}.json", row)
            completed += 1
            generated_tokens += len(sample.token_ids)
            if completed % 8 == 0 or not by_id:
                emit("generation_progress", output=str(out), completed=completed, total=len(job["requests"]),
                     tokens=generated_tokens, seconds=time.monotonic() - before)
    if by_id:
        raise RuntimeError(f"Engine ended before {len(by_id)} requests completed")
    atomic_json(out / "complete.json", {"requests": len(job["requests"]), "job_hash": job_hash})


if __name__ == "__main__":
    main()
