#!/usr/bin/env python3
"""Local, ordinary-model vLLM evaluation or calibration generation."""

import argparse
from collections import Counter
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

from steer_opd.math_data import grade_math, grade_rethinking_math, prompt_ids, read_rows
from steer_opd.rethinking_protocol import load_profile
from steer_opd.runtime import verify_runtime


ROOT = Path(__file__).resolve().parents[1]


def main():
    evaluation = load_profile()["evaluation"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--n", type=int, default=evaluation["n"])
    parser.add_argument("--max-tokens", type=int, default=evaluation["max_tokens"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=evaluation["temperature"])
    parser.add_argument("--top-p", type=float, default=evaluation["top_p"])
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--grader", choices=["rethinking", "math_verify"], default="rethinking")
    parser.add_argument("--correct-target", type=int, default=0)
    parser.add_argument("--correct-ids-from", type=Path)
    parser.add_argument("--skip-overlong", action="store_true",
                        help="Calibration only: record and exclude prompts over 1024 tokens")
    args = parser.parse_args()
    grader = grade_rethinking_math if args.grader == "rethinking" else grade_math
    verify_runtime()
    if not os.environ.get("STEER_OPD_GPU_ADMISSION") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("Launch through run_on_idle_gpus.py")
    out = args.output.resolve()
    if not out.is_relative_to(ROOT) or out.exists():
        raise ValueError("Output must be a new project-local file")
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.input)
    if args.correct_ids_from:
        selected = [r["id"] for r in read_rows(args.correct_ids_from)
                    if r["correct"] and r["finish_reason"] != "length"]
        order = {key: index for index, key in enumerate(dict.fromkeys(selected))}
        rows = sorted([r for r in rows if r["id"] in order], key=lambda r: order[r["id"]])
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        raise ValueError("No input questions")
    from vllm import LLM, SamplingParams

    started = time.monotonic()
    llm = LLM(model=args.model, tokenizer=args.model, dtype="bfloat16",
              trust_remote_code=False, tensor_parallel_size=1,
              gpu_memory_utilization=0.90, max_model_len=args.max_tokens + 1024,
              max_num_seqs=64, enforce_eager=True, enable_prefix_caching=False,
              generation_config="vllm", seed=args.seed)
    tokenizer = llm.get_tokenizer()
    encoded = [prompt_ids(tokenizer, r["question"], enable_thinking=args.enable_thinking) for r in rows]
    stop_token_ids = []
    for marker in evaluation["stop_markers"]:
        marker_ids = tokenizer.encode(marker, add_special_tokens=False)
        if len(marker_ids) == 1:
            stop_token_ids.append(marker_ids[0])
    overlong = [r["id"] for r, ids in zip(rows, encoded) if len(ids) > 1024]
    if overlong:
        if not args.skip_overlong:
            raise ValueError(f"Overlong prompts require explicit handling: {overlong}")
        pairs = [(r, ids) for r, ids in zip(rows, encoded) if len(ids) <= 1024]
        rows, encoded = [p[0] for p in pairs], [p[1] for p in pairs]
        if not rows:
            raise ValueError("No questions remain after overlong-prompt filtering")
    params = SamplingParams(n=args.n, temperature=args.temperature, top_p=args.top_p,
                            top_k=evaluation["top_k"], max_tokens=args.max_tokens,
                            stop_token_ids=stop_token_ids or None,
                            repetition_penalty=1.0)
    records, correct_complete_ids = [], set()
    metadata = {"started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "model": args.model, "input": str(args.input.resolve()),
                "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
                "n": args.n, "max_tokens": args.max_tokens, "seed": args.seed,
                "temperature": args.temperature, "top_p": args.top_p, "top_k": -1,
                "enable_thinking": args.enable_thinking, "grader": args.grader,
                "stop_token_ids": stop_token_ids, "request_seed": None,
                "steering": False, "overlong_excluded_ids": overlong,
                "gpu_admission": os.environ["STEER_OPD_GPU_ADMISSION"],
                "packages": {p: importlib.metadata.version(p) for p in ["torch", "transformers", "vllm", "math-verify"]}}
    with out.open("x") as stream:
        for offset in range(0, len(rows), args.batch_size):
            batch = rows[offset:offset + args.batch_size]
            requests = [{"prompt_token_ids": ids} for ids in encoded[offset:offset + len(batch)]]
            outputs = llm.generate(requests, params, use_tqdm=False)
            for row, result in zip(batch, outputs):
                for answer in result.outputs:
                    record = {"id": row["id"], "sample": answer.index, "answer": str(row["answer"]),
                              "text": answer.text, "token_ids": list(answer.token_ids),
                              "prompt_token_ids": list(result.prompt_token_ids),
                              "response_tokens": len(answer.token_ids), "finish_reason": answer.finish_reason,
                              **grader(row["answer"], answer.text)}
                    records.append(record)
                    if record["correct"] and record["finish_reason"] != "length":
                        correct_complete_ids.add(row["id"])
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            print(json.dumps({"completed_questions": min(offset + len(batch), len(rows)),
                              "samples": len(records), "accuracy": sum(r["correct"] for r in records) / len(records),
                              "correct_complete_questions": len(correct_complete_ids),
                              "elapsed_seconds": time.monotonic() - started}), flush=True)
            if args.correct_target and len(correct_complete_ids) >= args.correct_target:
                break
    metadata.update({"samples": len(records), "questions": len({r["id"] for r in records}),
                     "avg_accuracy": sum(r["correct"] for r in records) / len(records),
                     "truncation_rate": sum(r["finish_reason"] == "length" for r in records) / len(records),
                     "mean_response_tokens": sum(r["response_tokens"] for r in records) / len(records),
                     "grading_statuses": dict(Counter(r["grading_status"] for r in records)),
                     "elapsed_seconds": time.monotonic() - started,
                     "output_sha256": hashlib.sha256(out.read_bytes()).hexdigest()})
    out.with_suffix(".summary.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
