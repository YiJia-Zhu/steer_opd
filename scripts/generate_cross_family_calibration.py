#!/usr/bin/env python3
"""Cache native-template demonstrations and an independent dev competence check."""

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import time

from steer_opd.math_data import grade_math, read_rows
from steer_opd.runtime import verify_runtime


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--teacher-answers", type=Path)
    args = parser.parse_args()
    runtime = verify_runtime()
    if not os.environ.get("STEER_OPD_GPU_ADMISSION"):
        raise RuntimeError("Launch through run_on_idle_gpus.py")
    out = args.output_dir.resolve()
    if not out.is_relative_to(ROOT) or out.exists():
        raise ValueError("Use a new project-local output directory")
    config = json.loads(args.config.read_text())
    spec = config["teacher"] if args.model_key == "teacher" else next(
        s for s in config["students"] if s["key"] == args.model_key)
    protocol = config["steering_probe"]
    from vllm import LLM, SamplingParams
    from diagnose_cross_family_overlap import build_prompt

    model_path = Path(spec["path"])
    model_path = model_path if model_path.is_absolute() else ROOT / model_path
    started = time.monotonic()
    # The ~57 GiB teacher needs a larger fraction than the small-model pilot.
    llm = LLM(model=str(model_path), tokenizer=str(model_path), dtype="bfloat16",
              trust_remote_code=False, tensor_parallel_size=1, enforce_eager=True,
              gpu_memory_utilization=0.92 if args.model_key == "teacher" else 0.55,
              max_model_len=protocol["max_new_tokens"] + 1024,
              max_num_batched_tokens=4096,
              kernel_config={"moe_backend": "triton"},
              enable_prefix_caching=False, generation_config="vllm", seed=config["seed"])
    tokenizer = llm.get_tokenizer()
    calibration = read_rows(ROOT / protocol["calibration_path"])[:protocol["calibration_limit"]]
    if args.teacher_answers:
        valid = {r["id"] for r in read_rows(args.teacher_answers)
                 if r["correct"] and r["finish_reason"] != "length"}
        calibration = [r for r in calibration if r["id"] in valid][:protocol["pair_limit"]]
        if len(calibration) < protocol["minimum_pairs"]:
            raise ValueError(f"Only {len(calibration)} validated teacher demonstrations")
    dev = read_rows(ROOT / config["dev_path"])[:config["question_limit"]]
    if {r["id"] for r in calibration} & {r["id"] for r in dev}:
        raise ValueError("Calibration/dev leakage")
    out.mkdir(parents=True)
    summaries = {}
    params = SamplingParams(n=1, temperature=0.7, top_p=0.95, top_k=-1,
                            max_tokens=protocol["max_new_tokens"], seed=config["seed"],
                            repetition_penalty=1.0)
    for split, rows in [("calibration", calibration), ("dev", dev)]:
        encoded = [build_prompt(tokenizer, r["question"], spec) for r in rows]
        if any(len(ids) > 1024 for ids in encoded):
            raise ValueError("Prompt exceeds reserved 1024 tokens; do not silently truncate")
        records = []
        with (out / f"{split}.jsonl").open("x") as stream:
            results = llm.generate(
                [{"prompt_token_ids": ids} for ids in encoded],
                params, use_tqdm=False)
            for row, result in zip(rows, results):
                answer = result.outputs[0]
                record = {"id": row["id"], "question": row["question"], "answer": str(row["answer"]),
                          "text": answer.text, "response_token_ids": list(answer.token_ids),
                          "prompt_token_ids": list(result.prompt_token_ids),
                          "finish_reason": answer.finish_reason, "response_tokens": len(answer.token_ids),
                          **grade_math(row["answer"], answer.text)}
                records.append(record)
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            print(json.dumps({"model": args.model_key, "split": split,
                              "completed": len(records), "total": len(rows)}), flush=True)
        summaries[split] = {"questions": len(records), "correct": sum(r["correct"] for r in records),
                            "correct_complete": sum(r["correct"] and r["finish_reason"] != "length" for r in records),
                            "accuracy": sum(r["correct"] for r in records) / len(records),
                            "truncation_rate": sum(r["finish_reason"] == "length" for r in records) / len(records),
                            "mean_response_tokens": sum(r["response_tokens"] for r in records) / len(records),
                            "file_sha256": hashlib.sha256((out / f"{split}.jsonl").read_bytes()).hexdigest()}
    summary = {"spec": spec, "runtime": runtime, "protocol": protocol,
               "temperature": 0.7, "top_p": 0.95, "top_k": -1, "seed": config["seed"],
               "generation_backend": "vllm; Triton unquantized MoE backend",
               "teacher_answers": str(args.teacher_answers) if args.teacher_answers else None,
               "splits": summaries, "elapsed_seconds": time.monotonic() - started,
               "parameter_updates": 0, "steering": False,
               "gpu_admission": os.environ["STEER_OPD_GPU_ADMISSION"],
               "created_at": dt.datetime.now(dt.timezone.utc).isoformat()}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
