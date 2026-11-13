#!/usr/bin/env python3
"""Freeze disjoint local splits; shared datasets/models remain read-only."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import re

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from steer_opd.math_data import TASK_SUFFIX, prompt_ids, read_rows
from steer_opd.resumable_core import ROOT, atomic_json, digest


def normalized(text):
    text = text.removesuffix(TASK_SUFFIX)
    return re.sub(r"\s+", "", text).casefold()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/figure8_first_run.json")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    out = ROOT / cfg["data_dir"]
    out.mkdir(parents=True, exist_ok=True)
    signature = digest(cfg)
    if (out / "manifest.json").exists():
        assert json.loads((out / "manifest.json").read_text())["config_hash"] == signature
        print("Existing immutable splits verified")
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
