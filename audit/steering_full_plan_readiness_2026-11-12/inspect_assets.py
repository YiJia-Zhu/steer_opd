"""CPU-only, local-file inventory for the user's new research plan.

Run from the project root. Does not load weights, download, or initialize CUDA.
"""

import hashlib
import json
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import pyarrow.parquet as pq
from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).with_name("cpu_inventory.json")
NAMES = ["Qwen3-4B", "Qwen3-1.7B-Base", "Qwen3-0.6B"]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_merges(tokenizer_json):
    return [tuple(item.split(" ")) if isinstance(item, str) else tuple(item)
            for item in tokenizer_json["model"]["merges"]]


def main():
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "CPU metadata/tokenizer checks only; no weights loaded or hashed",
        "plan_sha256": sha256(ROOT / "docs/Steering_OPD_完整研究方案.md"),
        "gpu_tasks_launched": False,
        "downloads": False,
        "models": {},
        "comparisons": [],
        "packages": {},
    }
    raw = {}
    tokenizers = {}
    for name in NAMES:
        path = ROOT / "assets/models" / name
        config = json.loads((path / "config.json").read_text())
        tokenizer_config = json.loads((path / "tokenizer_config.json").read_text())
        generation_config = json.loads((path / "generation_config.json").read_text())
        raw[name] = json.loads((path / "tokenizer.json").read_text())
        tokenizers[name] = Tokenizer.from_file(str(path / "tokenizer.json"))
        weights = sorted(path.glob("*.safetensors"))
        vocab_ids = set(tokenizers[name].get_vocab().values())
        report["models"][name] = {
            "path": str(path.relative_to(ROOT)),
            "resolved_path": str(path.resolve()),
            "model_type": config["model_type"],
            "layers": config["num_hidden_layers"],
            "hidden_size": config["hidden_size"],
            "output_vocab_size": config["vocab_size"],
            "tokenizer_vocab_size": len(vocab_ids),
            "output_rows_without_tokenizer_entry": config["vocab_size"] - len(vocab_ids),
            "model_eos_id": config.get("eos_token_id"),
            "generation_defaults": {key: generation_config.get(key) for key in
                                    ["eos_token_id", "pad_token_id", "do_sample", "temperature", "top_p", "top_k"]},
            "tokenizer_eos": tokenizer_config.get("eos_token"),
            "tokenizer_pad": tokenizer_config.get("pad_token"),
            "chat_template_sha256": hashlib.sha256(str(tokenizer_config.get("chat_template")).encode()).hexdigest(),
            "config_sha256": sha256(path / "config.json"),
            "tokenizer_sha256": sha256(path / "tokenizer.json"),
            "weight_file_count": len(weights),
            "weight_bytes": sum(item.stat().st_size for item in weights),
        }
    samples = [
        "Solve 1 + 1.\nAnswer: 2", " leading space\n\tsecond line", "中文数学：二加二等于四。",
        r"Let $x^2=4$. Therefore \boxed{2}.", "unbelievable", "<think>\n</think>\n",
        "<|im_start|>assistant\n", "<tool_response>2</tool_response>",
    ]
    teacher_name = NAMES[0]
    teacher_vocab = tokenizers[teacher_name].get_vocab()
    for name in NAMES[1:]:
        other_vocab = tokenizers[name].get_vocab()
        shared = set(teacher_vocab) & set(other_vocab)
        report["comparisons"].append({
            "models": [teacher_name, name],
            "ordinary_vocab_equal": raw[teacher_name]["model"]["vocab"] == raw[name]["model"]["vocab"],
            "ordinary_vocab_size": len(raw[name]["model"]["vocab"]),
            "normalized_merges_equal": normalized_merges(raw[teacher_name]) == normalized_merges(raw[name]),
            "normalizer_equal": raw[teacher_name].get("normalizer") == raw[name].get("normalizer"),
            "pre_tokenizer_equal": raw[teacher_name].get("pre_tokenizer") == raw[name].get("pre_tokenizer"),
            "decoder_equal": raw[teacher_name].get("decoder") == raw[name].get("decoder"),
            "shared_tokens": len(shared),
            "shared_id_conflicts": [(token, teacher_vocab[token], other_vocab[token])
                                    for token in sorted(shared) if teacher_vocab[token] != other_vocab[token]],
            "teacher_only_tokens": {token: teacher_vocab[token] for token in sorted(set(teacher_vocab) - set(other_vocab))},
            "other_only_tokens": {token: other_vocab[token] for token in sorted(set(other_vocab) - set(teacher_vocab))},
            "sample_encodings": [{"text": sample,
                                  "teacher_ids": tokenizers[teacher_name].encode(sample, add_special_tokens=False).ids,
                                  "other_ids": tokenizers[name].encode(sample, add_special_tokens=False).ids}
                                 for sample in samples],
        })
    for package in ["torch", "transformers", "vllm", "verl", "pyarrow", "pytest", "tokenizers"]:
        try:
            report["packages"][package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            report["packages"][package] = None
    opd_path = ROOT / "ref/ref_rep/Rethinking-OPD/datasets/OpenThoughts3_opd.parquet"
    parquet = pq.ParquetFile(opd_path)
    report["data"] = {
        "opd_path": str(opd_path.relative_to(ROOT)),
        "opd_rows": parquet.metadata.num_rows,
        "opd_columns": parquet.schema.names,
        "aime2025": {item.name: sum(bool(line.strip()) for line in item.read_text().splitlines())
                     for item in sorted((ROOT / "assets/data/AIME2025").glob("*.jsonl"))},
        "math500_path_exists": (ROOT / "assets/data/MATH-500").exists(),
        "new_disjoint_split_created": False,
    }
    report["known_package_issue"] = {"runs_readme_exists": (ROOT / "runs/README.md").exists()}
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(OUTPUT.relative_to(ROOT)), "models": NAMES,
                      "opd_rows": report["data"]["opd_rows"],
                      "shared_token_counts": [row["shared_tokens"] for row in report["comparisons"]]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
