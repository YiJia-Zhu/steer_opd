#!/usr/bin/env python3
"""Read-only hashes of current-pair weights, tokenizers, and frozen splits."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def fingerprint(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            sha.update(block)
    return {"bytes": path.stat().st_size, "sha256": sha.hexdigest()}


def freeze():
    models = {}
    for name in ("Qwen3-1.7B-Base", "Qwen3-4B", "Qwen3-0.6B"):
        directory = ROOT / "assets/models" / name
        weights = sorted(directory.glob("*.safetensors"))
        if not weights:
            raise ValueError(f"Missing actual weights: {name}")
        files = weights + [path for path in directory.glob("*.json") if "generation" not in path.name]
        models[name] = {"resolved_path": str(directory.resolve()),
                        "files": {path.name: fingerprint(path) for path in sorted(set(files))}}
    data = ROOT / "data/figure8_shared_top16_v2_20261112"
    splits = {name: fingerprint(data/f"{name}.json") for name in ("opd", "direction", "dev", "aime2025")}
    if json.loads((data/"opd.json").read_text()) != json.loads((data/"direction.json").read_text()):
        raise ValueError("SV/OPD candidate questions differ")
    value = {"models": models, "splits": splits, "inputs_only_no_model_load_or_quality_claim": True}
    output = ROOT / "audit/full_scheme_2026-11-12/input_fingerprints.json"
    if output.exists() and json.loads(output.read_text()) != value:
        raise ValueError("Shared input fingerprint changed; use a new run version")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, ensure_ascii=False)+"\n")
    print(f"Frozen input hashes: {output}")


if __name__ == "__main__":
    freeze()
