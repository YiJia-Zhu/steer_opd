#!/usr/bin/env python3
"""Freeze versioned local splits; shared datasets/models remain read-only."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import re

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from steer_opd.math_data import TASK_SUFFIX, prompt_ids, read_rows
from steer_opd.resumable_core import ROOT, atomic_json, digest, read_json
from steer_opd.generation_runtime import checkpoint_config_hash


def normalized(text):
    text = text.removesuffix(TASK_SUFFIX)
    return re.sub(r"\s+", "", text).casefold()


def same_as_opd_splits(source, cfg):
    """Keep existing train/dev/test rows byte-equivalent after JSON decoding."""
    source = Path(source)
    previous = read_json(source / "manifest.json")
    if previous["source"] != cfg["source"]:
        raise ValueError("Reuse source does not match frozen OT3 source")
    splits = {name: read_json(source / f"{name}.json") for name in ["opd", "dev", "aime2025"]}
    for name, size in [("opd", cfg["train_questions"]), ("dev", cfg["dev_questions"])]:
        if len(splits[name]) != size or digest(splits[name]) != previous["splits"][name]["hash"]:
            raise ValueError(f"Reused {name} split failed frozen hash/count check")
    if cfg["direction_questions"] != cfg["train_questions"]:
        raise ValueError("same_as_opd requires identical direction/train question counts")
    sets = {}
    id_sets = {}
    for name, rows in splits.items():
        sets[name] = {normalized(row["question"]) for row in rows}
        id_sets[name] = {row["id"] for row in rows}
        if len(sets[name]) != len(rows) or len(id_sets[name]) != len(rows):
            raise ValueError(f"Duplicate reused {name} question")
    for a, b in [("opd", "dev"), ("opd", "aime2025"), ("dev", "aime2025")]:
        if sets[a] & sets[b] or id_sets[a] & id_sets[b]:
            raise ValueError(f"Reused {a}/{b} overlap")
    if len(splits["aime2025"]) != 30:
        raise ValueError("Expected AIME2025 30 questions")
    splits["direction"] = splits["opd"]
    return splits, previous


def prepare_same_pool(out, cfg, signature):
    source = ROOT / cfg["reuse_question_splits_from"]
    splits, previous = same_as_opd_splits(source, cfg)
    if cfg.get("common_training_ids_file"):
        common = read_json(ROOT / cfg["common_training_ids_file"])
        ids = set(common["valid_ids"])
        splits["opd"] = [row for row in splits["opd"] if row["id"] in ids]
        splits["direction"] = splits["opd"]
        if {row["id"] for row in splits["opd"]} != ids:
            raise ValueError("Common reference filter contains unknown training questions")
    manifest = {"config_hash": signature, "source": cfg["source"],
                "direction_data_policy": "same_as_opd", "reuse_question_splits_from": str(source.relative_to(ROOT)),
                "reused_manifest_hash": digest(previous), "counts": previous["counts"],
                "eligible": previous["eligible"], "prompt_examples": previous["prompt_examples"],
                "test_count": len(splits["aime2025"]), "splits": {},
                "intentional_opd_direction_overlap": len(splits["opd"]),
                "dev_test_isolation": "IDs and normalized question text checked",
                "dedup_scope": previous["dedup_scope"]}
    if cfg.get("common_training_ids_file"):
        manifest["common_reference_filter"] = cfg["common_training_ids_file"]
        manifest["candidate_training_questions"] = cfg["train_questions"]
        manifest["effective_training_questions"] = len(splits["opd"])
    for name, rows in splits.items():
        atomic_json(out / f"{name}.json", rows)
        manifest["splits"][name] = {"count": len(rows), "hash": digest(rows), "ids": [r["id"] for r in rows]}
    if cfg.get("additional_evaluation_sets") or cfg.get("dev_eval_steps"):
        # Exclude every prefix-tuning question from subsequent dev selection.
        fixed_ids = {row["id"] for row in splits["dev"][:cfg["diagnostic_questions"]]}
        selection_dev = [row for row in splits["dev"] if row["id"] not in fixed_ids]
        atomic_json(out / "selection_dev.json", selection_dev)
        manifest["splits"]["selection_dev"] = {"count": len(selection_dev), "hash": digest(selection_dev)}
        if "math500" in cfg.get("additional_evaluation_sets", []):
            tokenizer = AutoTokenizer.from_pretrained(ROOT / cfg["student"], local_files_only=True)
            excluded = {normalized(row["question"]) for row in splits["opd"] + splits["direction"]}
            if cfg.get("direction_questions_file"):
                excluded.update(normalized(row["question"]) for row in read_json(ROOT / cfg["direction_questions_file"]))
            tests, seen = [], set()
            for file in sorted((ROOT / "assets/data/MATH-500").rglob("*.parquet")):
                for index, row in enumerate(pq.read_table(file).to_pylist()):
                    question = row.get("problem", row.get("question"))
                    key = normalized(question)
                    if key in excluded:
                        raise ValueError("MATH500 test question overlaps frozen training/reference pool")
                    if key in seen:
                        continue
                    seen.add(key)
                    tests.append({"id": f"math500-{len(tests):04d}", "question": question,
                        "answer": str(row["answer"]), "student_prompt_ids": prompt_ids(tokenizer, question, enable_thinking=False)})
            if len(tests) != 500:
                raise ValueError(f"Expected 500 unique MATH500 tests, found {len(tests)}")
            atomic_json(out / "math500.json", tests)
            manifest["splits"]["math500"] = {"count": 500, "hash": digest(tests)}
    atomic_json(out / "manifest.json", manifest)
    print(json.dumps({"output": str(out), "direction_data_policy": "same_as_opd",
                      "split_sizes": {name: len(rows) for name, rows in splits.items()}}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/figure8_first_run.json")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    out = ROOT / cfg["data_dir"]
    out.mkdir(parents=True, exist_ok=True)
    signature = checkpoint_config_hash(cfg)
    if (out / "manifest.json").exists():
        manifest = read_json(out / "manifest.json")
        assert manifest["config_hash"] == signature
        for name, info in manifest["splits"].items():
            if digest(read_json(out / f"{name}.json")) != info["hash"]:
                raise ValueError(f"Frozen {name} split changed")
        print("Existing immutable splits verified")
        return
    if cfg.get("direction_data_policy") == "same_as_opd":
        prepare_same_pool(out, cfg, signature)
        return
    tokenizers = {role: AutoTokenizer.from_pretrained(ROOT / cfg[role], local_files_only=True)
                  for role in ["student", "teacher"]}
    tests = []
    for file in sorted((ROOT / "assets/data/AIME2025").glob("*.jsonl")):
        for index, row in enumerate(read_rows(file)):
            tests.append({"id": f"{file.stem}-{index:02d}", "question": row["question"],
                          "answer": str(row["answer"]), "source": str(file.relative_to(ROOT))})
    excluded = {normalized(r["question"]) for r in tests}
    # Exclude frozen MATH-500 questions when available, without adding an eval run.
    math_root = ROOT / "assets/data/MATH-500"
    for file in math_root.rglob("*.parquet"):
        for row in pq.read_table(file).to_pylist():
            question = row.get("problem", row.get("question"))
            if isinstance(question, str):
                excluded.add(normalized(question))
    seen, eligible = set(), []
    counts = {"source": 0, "duplicate": 0, "test_overlap": 0, "overlong": 0}
    for index, row in enumerate(pq.read_table(ROOT / cfg["source"]).to_pylist()):
        counts["source"] += 1
        text = row["prompt"][0]["content"]
        question = text.removesuffix(TASK_SUFFIX)
        key = normalized(question)
        if key in excluded:
            counts["test_overlap"] += 1
            continue
        if key in seen:
            counts["duplicate"] += 1
            continue
        seen.add(key)
        encoded = {role: prompt_ids(t, question, enable_thinking=False)
                   for role, t in tokenizers.items()}
        if max(map(len, encoded.values())) > cfg["max_prompt_tokens"]:
            counts["overlong"] += 1
            continue
        eligible.append({"id": f"ot3-{index:05d}", "source_row": index, "question": question,
                         "answer": str(row["reward_model"]["ground_truth"]),
                         "question_hash": hashlib.sha256(key.encode()).hexdigest(),
                         "student_prompt_ids": encoded["student"], "teacher_prompt_ids": encoded["teacher"]})
    random.Random(cfg["seed"]).shuffle(eligible)
    cursor = 0
    manifest = {"config_hash": signature, "source": cfg["source"], "counts": counts,
                "eligible": len(eligible), "splits": {}}
    for name, size in [("opd", cfg["train_questions"]), ("direction", cfg["direction_questions"]),
                       ("dev", cfg["dev_questions"])]:
        rows = eligible[cursor:cursor + size]
        assert len(rows) == size
        cursor += size
        atomic_json(out / f"{name}.json", rows)
        manifest["splits"][name] = {"count": size, "hash": digest(rows), "ids": [r["id"] for r in rows]}
    for row in tests:
        row["student_prompt_ids"] = prompt_ids(tokenizers["student"], row["question"], enable_thinking=False)
    atomic_json(out / "aime2025.json", tests)
    manifest["test_count"] = len(tests)
    manifest["dedup_scope"] = "exact normalized text; no claim of complete semantic decontamination"
    manifest["prompt_examples"] = {role: tokenizers[role].decode(eligible[0][f"{role}_prompt_ids"])
                                   for role in tokenizers}
    atomic_json(out / "manifest.json", manifest)
    print(json.dumps({"output": str(out), "counts": counts, "split_sizes": [100, 256, 128]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
