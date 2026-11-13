#!/usr/bin/env python3
"""Derive common-filtered text controls on CPU; no new model generations."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from transformers import AutoTokenizer

from steer_opd.reference_controls import common_records
from steer_opd.resumable_core import ROOT, atomic_json, digest, read_json


def grade(item):
    import signal
    from steer_opd.math_data import grade_rethinking_math
    def expired(*args):
        raise TimeoutError("Reference grading timeout")
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(10)
    try:
        return {"correct": grade_rethinking_math(item[0], item[1])["correct"], "status": "graded"}
    except Exception as exc:
        return {"correct": None, "status": "unverified", "error": type(exc).__name__}
    finally:
        signal.alarm(0)


def prepare(job):
    tokenizer = AutoTokenizer.from_pretrained(ROOT / job["student"], local_files_only=True)
    questions = read_json(ROOT / job["questions"])
    originals = {row["id"]: {role: read_json(ROOT / directory / f"{row['id']}.json")
                            for role, directory in job["roles"].items()} for row in questions}
    correctness, grading = {}, {}
    if job.get("both_correct"):
        inputs = [(row["answer"], originals[row["id"]][role]["text"])
                  for row in questions for role in ("teacher", "reference")]
        with ProcessPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(grade, inputs))
        grading = {row["id"]: dict(zip(("teacher", "reference"), results[2*i:2*i+2])) for i, row in enumerate(questions)}
        correctness = {qid: {role: record["correct"] is True for role, record in roles.items()} for qid, roles in grading.items()}
    output = ROOT / job["output"]
    kept, rejected = [], {}
    for row in questions:
        records, reason = common_records(originals[row["id"]], tokenizer,
            normalize_template=job.get("normalize_template", False), match_length=job.get("match_length", False),
            correctness=correctness[row["id"]] if correctness else None)
        if reason:
            rejected[row["id"]] = reason
            continue
        kept.append(row)
        for role, record in records.items():
            atomic_json(output / role / f"{row['id']}.json", record)
    if len(kept) < job.get("minimum_pairs", 1):
        raise ValueError("Insufficient common usable T/R/S reference questions")
    atomic_json(output / "questions.json", kept)
    atomic_json(output / "manifest.json", {"job": job, "job_hash": digest(job), "valid_ids": [r["id"] for r in kept],
        "rejected": rejected, "correctness": correctness, "grading": grading,
        "source_hash": digest(originals), "parameter_updates": 0})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    prepare(read_json(parser.parse_args().job))
