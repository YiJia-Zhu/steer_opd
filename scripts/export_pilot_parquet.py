#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/math500_pilot_v2"


def main():
    records = {}
    for name in ["train_pilot", "dev_pilot", "train", "dev"]:
        rows = [json.loads(line) for line in (DATA / f"{name}.jsonl").read_text().splitlines()]
        converted = [{
            "data_source": "steer_opd/dapo_snapshot",
            "prompt": [{"role": "user", "content": row["question"] +
                        " Please reason step by step, and put your final answer within \\boxed{}."}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": str(row["answer"])},
            "extra_info": {"id": row["id"], "split": name, "question_sha256": row["question_sha256"]},
        } for row in rows]
        path = DATA / f"{name}.parquet"
        pq.write_table(pa.Table.from_pylist(converted), path)
        assert pq.read_table(path).to_pylist() == converted
        records[name] = {"rows": len(rows), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (DATA / "parquet_manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
