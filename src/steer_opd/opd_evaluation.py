"""Bare-student AIME25 sampling, raw output retention and math-verify scoring."""
from collections import defaultdict
import importlib.metadata
import json
from pathlib import Path

from steer_opd.direction_bank import file_sha256
from steer_opd.resumable_core import atomic_json, digest, read_json


def aime_questions(directory):
    directory = Path(directory)
    questions = []
    sources = {}
    for part in ("I", "II"):
        path = directory / f"aime2025-{part}.jsonl"
        sources[path.name] = file_sha256(path)
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(rows) != 15:
            raise ValueError(f"Expected 15 AIME2025-{part} questions, found {len(rows)}")
        for index, row in enumerate(rows, 1):
            if not row.get("question") or not str(row.get("answer", "")).strip():
                raise ValueError("AIME question or gold answer is missing")
            questions.append({"id": f"AIME2025-{part}-{index:02d}",
                              "question": row["question"], "answer": str(row["answer"])})
    if len({q["question"] for q in questions}) != 30:
        raise ValueError("Duplicate questions in AIME2025")
    return questions, sources


def grade_answer(text, gold):
    """Use the installed verifier's extraction and equivalence rules directly."""
    from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify
    expected = parse(gold, extraction_config=[ExprExtractionConfig()],
                     extraction_mode="first_match", parsing_timeout=5)
    if not expected:
        raise ValueError("math-verify could not parse the gold answer")
    extracted = parse(text, extraction_config=[LatexExtractionConfig(boxed_match_priority=0),
                                              ExprExtractionConfig()],
                      extraction_mode="first_match", parsing_timeout=5)
    return {"correct": bool(extracted and verify(expected, extracted, timeout_seconds=5)),
            "extracted_answer": [str(value) for value in extracted],
            "answer_extracted": bool(extracted)}


def score_outputs(questions, outputs, *, samples_per_question=16, grader=grade_answer):
    """acc is mean sample correctness (estimated pass@1); pass@16 is any hit."""
    expected = {row["id"]: row for row in questions}
    if len(expected) != len(questions) or not expected or samples_per_question != 16:
        raise ValueError("Need distinct questions and exactly 16 samples per question")
    grouped = defaultdict(list)
    seen = set()
    scored = []
    for row in outputs:
        key = (row["question_id"], row["sample_index"])
        if (key in seen or key[0] not in expected or type(key[1]) is not int
                or not 0 <= key[1] < samples_per_question):
            raise ValueError("Unexpected or duplicate evaluation sample")
        seen.add(key)
        result = {**row, **grader(row["text"], expected[key[0]]["answer"])}
        scored.append(result)
        grouped[key[0]].append(result)
    if set(grouped) != set(expected) or any(len(rows) != samples_per_question for rows in grouped.values()):
        raise ValueError("Incomplete evaluation: metrics require all 16 samples for every question")
    per_question = []
    for question in questions:
        rows = grouped[question["id"]]
        correct = sum(row["correct"] for row in rows)
        per_question.append({"question_id": question["id"], "gold_answer": question["answer"],
                             "samples": samples_per_question, "correct": correct,
                             "acc": correct / samples_per_question, "pass@16": float(correct > 0)})
    summary = {"questions": len(questions), "samples_per_question": samples_per_question,
               "samples": len(scored), "correct_samples": sum(row["correct"] for row in scored),
               "acc": sum(row["correct"] for row in scored) / len(scored),
               "pass@16": sum(row["pass@16"] for row in per_question) / len(per_question),
               "acc_definition": "mean correctness over all 16 samples; estimated pass@1, not greedy accuracy",
               "pass@16_definition": "fraction of questions with at least one correct answer among 16 samples",
               "answer_extracted_samples": sum(row["answer_extracted"] for row in scored),
               "length_finished_samples": sum(row.get("finish_reason") == "length" for row in scored),
               "empty_samples": sum(not row["text"].strip() for row in scored),
               "verifier": "math-verify", "verifier_version": importlib.metadata.version("math-verify")}
    return summary, per_question, scored


def pack_outputs(questions, prompts, outputs, *, n=16):
    """Retain the returned texts, tokens and completion metadata before grading."""
    if len(outputs) != len(questions) or len(prompts) != len(questions):
        raise ValueError("vLLM returned an incomplete AIME batch")
    rows = []
    for question, prompt, output in zip(questions, prompts, outputs):
        if list(output.prompt_token_ids) != prompt or len(output.outputs) != n:
            raise ValueError("vLLM changed the prompt or returned fewer than 16 samples")
        indices = set()
        for candidate in output.outputs:
            if candidate.index in indices or candidate.index not in range(n):
                raise ValueError("vLLM returned unexpected completion indices")
            indices.add(candidate.index)
            rows.append({"question_id": question["id"], "sample_index": candidate.index,
                         "request_id": str(output.request_id), "prompt_token_ids": prompt,
                         "text": candidate.text, "token_ids": list(candidate.token_ids),
                         "cumulative_logprob": candidate.cumulative_logprob,
                         "finish_reason": candidate.finish_reason, "stop_reason": candidate.stop_reason})
    return rows


def run_evaluation(job):
    from steer_opd.checkpoint_identity import checkpoint_content_id
    from steer_opd.execution_policy import require_execution_allowed
    from steer_opd.experiment_runtime import StageCheckpoint
    from steer_opd.math_data import prompt_ids
    from steer_opd.stage_io import project

    require_execution_allowed(job)
    from steer_opd.generation_policy import DEFAULT_MAX_GENERATION_LENGTH,presence_penalty
    # Canonicalize the effective generation setting into the stage identity.
    job={**job,'generation':{**job['generation'],
         'max_tokens':job['generation'].get('max_tokens',DEFAULT_MAX_GENERATION_LENGTH),
         'presence_penalty':presence_penalty(job['generation'])}}
    output = project(job["output"])
    model = project(job["model"])
    if "step_5_eval" not in output.parts:
        raise ValueError("Evaluation outputs must follow the step_5_eval layout")
    generation = job["generation"]
    if (generation["n"] != 16 or job.get("seed") != 42 or generation["temperature"] <= 0
            or not 0 < generation["top_p"] <= 1 or generation["max_tokens"] < 1):
        raise ValueError("AIME25 evaluation requires seed42 and 16 stochastic samples per question")
    questions, dataset_sources = aime_questions(project(job["dataset_dir"]))
    checkpoint = checkpoint_content_id(model)
    from steer_opd import math_data,generation_policy
    identity = {"job": job, "checkpoint_content_id": checkpoint, "dataset_sha256": dataset_sources,
                "vllm_version": importlib.metadata.version("vllm"), "steering": False,
                "source_sha256": {"opd_evaluation.py": file_sha256(Path(__file__)),
                                  "generation_policy.py": file_sha256(Path(generation_policy.__file__)),
                                  "math_data.py": file_sha256(Path(math_data.__file__))}}
    binding = digest(identity)
    sampling = StageCheckpoint(output / "sampling", binding)
    raw_path = sampling.output / "vllm_outputs.json"
    if not sampling.load():
        from steer_opd.experiment_runtime import configure_seed, engine_startup_guard, require_artifact_space
        from steer_opd.resumable_core import require_gpu
        from transformers import AutoTokenizer
        configure_seed(42)
        require_gpu()
        require_artifact_space(output, 512 * 1024**2)
        tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
        prompts = [prompt_ids(tokenizer, row["question"], job.get("enable_thinking", False)) for row in questions]
        max_model_len = max(map(len, prompts)) + generation["max_tokens"]
        import os
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        os.environ["VLLM_BATCH_INVARIANT"] = "1"
        from vllm import LLM, SamplingParams
        with engine_startup_guard():
            llm = LLM(model=str(model), dtype="bfloat16", tensor_parallel_size=1, seed=42,
                      max_model_len=max_model_len, gpu_memory_utilization=job.get("gpu_memory_utilization", .75),
                      generation_config="vllm", hf_overrides={"head_dtype": "float32"})
        params = SamplingParams(n=16, seed=42, temperature=generation["temperature"],
                                top_p=generation["top_p"], top_k=-1, max_tokens=generation["max_tokens"],
                                presence_penalty=generation['presence_penalty'])
        # A single n=16 request per question supplies distinct random draws.
        # Repeating identical seed42 requests 16 times would repeat responses.
        outputs = llm.generate([{"prompt_token_ids": ids} for ids in prompts], params, use_tqdm=True)
        raw = pack_outputs(questions, prompts, outputs)
        atomic_json(raw_path, raw)
        atomic_json(sampling.output / "identity.json", identity)
        atomic_json(sampling.output / "questions.json", questions)
        sampling.commit([raw_path, sampling.output / "identity.json", sampling.output / "questions.json"],
                        samples=len(raw), questions=len(questions), steering=False)
    raw = read_json(raw_path)
    scoring = StageCheckpoint(output / "scoring", digest({"sampling": binding,
        "raw_sha256": file_sha256(raw_path), "verifier_version": importlib.metadata.version("math-verify"),
        "scoring_source_sha256": file_sha256(Path(__file__)), "schema": "aime25_math_verify_v1"}))
    if scoring.load():
        return read_json(scoring.output / "metrics.json")
    metrics, per_question, scored = score_outputs(questions, raw)
    files = []
    for name, value in (("metrics.json", metrics), ("per_question.json", per_question), ("scored_outputs.json", scored)):
        path = scoring.output / name
        atomic_json(path, value)
        files.append(path)
    scoring.commit(files, samples=metrics["samples"], steering=False)
    return metrics
