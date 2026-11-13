#!/usr/bin/env python3
"""Isolated throughput/profiler probe; no campaign data or checkpoints mutated."""

import argparse
import os
from pathlib import Path
import statistics
import time

from steer_opd.resumable_core import ROOT, atomic_json, read_json, require_gpu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = (ROOT / args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    require_gpu()
    os.environ.update({"VLLM_USE_V2_MODEL_RUNNER": "0", "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                       "VLLM_USE_FLASHINFER_SAMPLER": "0", "OMP_NUM_THREADS": "1",
                       "STEER_OPD_VLLM_SUPPORT": "151665", "STEER_OPD_REFERENCE_OPS": "0"})
    for key in ["STEER_OPD_VLLM_VECTOR", "STEER_OPD_CAPTURE_DIR", "STEER_OPD_REPLAY_AUDIT_TOKENS"]:
        os.environ.pop(key, None)
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import RequestOutputKind

    before = time.monotonic()
    llm = LLM(
        model=str(ROOT / "assets/models/Qwen3-1.7B-Base"), dtype="bfloat16",
        tensor_parallel_size=1, gpu_memory_utilization=0.9, max_model_len=32024,
        enable_prefix_caching=False, enable_chunked_prefill=False,
        generation_config="vllm", seed=42, enforce_eager=False,
        hf_overrides={"head_dtype": "float32"},
        worker_cls="steer_opd.vllm_worker.SteeringWorker",
        compilation_config={"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"},
        profiler_config={"profiler": "torch", "torch_profiler_dir": str(out / "traces"),
                         "torch_profiler_with_stack": False, "torch_profiler_record_shapes": False,
                         "torch_profiler_dump_cuda_time_total": True, "ignore_frontend": True},
    )
    startup = time.monotonic() - before
    run = ROOT / "runs/figure8_first_run_20261112"
    source_job = read_json(run / "jobs/fixed_initial_prefixes.json")
    texts = [read_json(ROOT / source_job["output"] / f"{r['key']}.json") for r in source_job["requests"]]
    texts = [r for r in texts if len(r["token_ids"]) >= 100]
    results = []
    for length in [128, 8192, 16384, 30000]:
        prompts = []
        for i in range(16):
            row = texts[i % len(texts)]
            tokens = row["prompt_token_ids"] + row["token_ids"] * 320
            prompts.append({"prompt_token_ids": tokens[:length]})
        params = [SamplingParams(temperature=0.7, top_p=0.95, top_k=-1, seed=42 + i,
                                 max_tokens=512, min_tokens=512, ignore_eos=True,
                                 logprobs=None, output_kind=RequestOutputKind.FINAL_ONLY)
                  for i in range(16)]
        llm.enqueue(prompts, params, use_tqdm=False)
        started = time.monotonic()
        durations, outputs = [], []
        while llm.llm_engine.has_unfinished_requests():
            if length == 16384 and len(durations) == 32:
                llm.start_profile(profile_prefix="decode_16k", max_iterations=32)
            tick = time.monotonic()
            outputs.extend(x for x in llm.llm_engine.step() if x.finished)
            durations.append(time.monotonic() - tick)
        elapsed = time.monotonic() - started
        if length == 16384:
            llm.stop_profile()
        total = sum(len(sample.token_ids) for row in outputs for sample in row.outputs)
        report = {"prefix_length": length, "responses": len(outputs), "generated_tokens": total,
                  "seconds_including_prefill": elapsed, "tokens_per_second_including_prefill": total / elapsed,
                  "last_64_client_step_median_ms": statistics.median(durations[-64:]) * 1000,
                  "engine_calls": len(durations)}
        results.append(report)
        print(report, flush=True)
        atomic_json(out / "results.json", {"startup_seconds": startup, "cases": results,
                    "note": "Dev-derived repeated prefixes, fixed 512 outputs/request, 16 requests; hardware probe only; not accuracy evaluation. Profiled 16k timing includes profiler overhead."})


if __name__ == "__main__":
    main()
