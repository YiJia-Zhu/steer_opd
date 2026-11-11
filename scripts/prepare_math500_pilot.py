#!/usr/bin/env python3
"""Prepare isolated DAPO splits and audit leakage against MATH-500.

Reads the paper's Parquet with pyarrow. Original assets are read-only.
Near-duplicate exclusion is conservative lexical screening, not a proof that
the source data or pretrained models are free of semantic contamination.
"""

import argparse
from collections import Counter, defaultdict
import datetime as dt
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "ref/ref_rep/Rethinking-OPD/datasets/dapo-math-17k.parquet"
TEST = Path("/home/zhenning/data/huggingface_datasets/MATH-500/test.jsonl")


def normalize(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def shingles(text):
    tokens = re.findall(r"[a-z]+|\d+|[^\w\s]", normalize(text).lower())
    return {tuple(tokens[i:i + 5]) for i in range(max(len(tokens) - 4, 0))}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path, rows):
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "data/math500_pilot_v2")
    args = parser.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to(ROOT):
        raise ValueError("Output must stay in the project")
    if (out / "split_manifest.json").exists():
        raise FileExistsError("Existing split manifest: choose a new version to avoid silent changes")
    out.mkdir(parents=True, exist_ok=True)
    import pyarrow.parquet as pq

    prefix = ("Solve the following math problem step by step. The last line of your response should be "
              "of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n")
    suffix = '\n\nRemember to put your answer on its own line after "Answer:".'
    raw = []
    for source_row in pq.read_table(SOURCE).to_pylist():
        text = source_row["prompt"][0]["content"]
        if not text.startswith(prefix) or not text.endswith(suffix):
            raise ValueError("Unrecognized source prompt wrapper; do not strip heuristically")
        raw.append({"id": str(source_row["extra_info"]["index"]),
                    "prompt": text[len(prefix):-len(suffix)],
                    "answer": str(source_row["reward_model"]["ground_truth"]),
                    "source": source_row["data_source"]})
    test = [json.loads(line) for line in TEST.read_text().splitlines() if line.strip()]
    assert len(test) == len({r["unique_id"] for r in test}) == 500
    test_by_text = {normalize(r["problem"]): r for r in test}
    groups = defaultdict(list)
    for row in raw:
        groups[normalize(row["prompt"])].append(row)
    test_shingles = [shingles(r["problem"]) for r in test]
    inverted = defaultdict(set)
    for index, parts in enumerate(test_shingles):
        for part in parts:
            inverted[part].add(index)

    kept, excluded, merged = [], [], []
    for text, rows in groups.items():
        ids = [r["id"] for r in rows]
        answers = sorted({normalize(str(r["answer"])) for r in rows})
        if len(answers) != 1:
            excluded.append({"ids": ids, "reason": "conflicting_answers", "answers": answers})
            continue
        if text in test_by_text:
            excluded.append({"ids": ids, "reason": "exact_test_overlap",
                             "test_id": test_by_text[text]["unique_id"]})
            continue
        parts = shingles(text)
        candidates = Counter(index for part in parts for index in inverted.get(part, ()))
        matches = []
        for index, intersection in candidates.items():
            other = test_shingles[index]
            shortest = min(len(parts), len(other))
            jaccard = intersection / max(len(parts | other), 1)
            containment = intersection / max(shortest, 1)
            if shortest >= 8 and (jaccard >= 0.8 or containment >= 0.9):
                matches.append({"test_id": test[index]["unique_id"],
                                "jaccard": jaccard, "containment": containment})
        if matches:
            excluded.append({"ids": ids, "reason": "lexical_near_test_overlap", "matches": matches})
            continue
        row = rows[0]
        kept.append({"id": row["id"], "question": row["prompt"], "answer": row["answer"],
                     "source": row["source"], "source_ids": ids,
                     "question_sha256": hashlib.sha256(text.encode()).hexdigest()})
        if len(rows) > 1:
            merged.append({"kept": row["id"], "ids": ids})

    random.Random(42).shuffle(kept)
    assert len(kept) > 1024 + 256 + 2048
    calibration, dev, train = kept[:1024], kept[1024:1280], kept[1280:]
    sets = [{r["question_sha256"] for r in split} for split in [calibration, dev, train]]
    assert all(not sets[i] & sets[j] for i in range(3) for j in range(i))
    splits = {"calibration_candidates": calibration, "dev": dev, "train": train,
              "dev_pilot": dev[:128], "train_pilot": train[:2048]}
    for name, rows in splits.items():
        write_jsonl(out / f"{name}.jsonl", rows)
    write_jsonl(out / "math500.jsonl", [
        {"id": row["unique_id"], "question": row["problem"], "answer": row["answer"],
         "subject": row["subject"], "level": row["level"]} for row in test
    ])
    write_jsonl(out / "exclusions.jsonl", excluded)
    write_jsonl(out / "merged_duplicates.jsonl", merged)
    manifest = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(), "seed": 42,
        "source": {"path": str(SOURCE), "sha256": digest(SOURCE), "rows": len(raw),
                   "git_commit": "ac26e38d6f1572eb027597b48a9f4e01f6915ef8"},
        "test": {"path": str(TEST), "sha256": digest(TEST), "rows": 500},
        "normalization": "Unicode NFKC, collapse whitespace, strip",
        "near_duplicate_rule": "casefold lexical 5-grams; >=8 distinct shorter shingles; Jaccard>=0.8 OR containment>=0.9",
        "normalized_unique_source_questions": len(groups), "retained_unique_questions": len(kept),
        "exclusion_groups_by_reason": dict(Counter(x["reason"] for x in excluded)),
        "split_counts": {name: len(rows) for name, rows in splits.items()},
        "files": {p.name: {"sha256": digest(p), "bytes": p.stat().st_size}
                  for p in sorted(out.glob("*.jsonl"))},
        "script_sha256": digest(Path(__file__)),
        "limitations": ["Lexical exclusion does not establish absence of semantic contamination.",
                        "Paper training Parquet is reused after test exclusions, deduplication and new splits.",
                        "Base-model pretraining or teacher RL overlap is not audited."],
    }
    (out / "split_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in ["retained_unique_questions",
                     "exclusion_groups_by_reason", "split_counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
