"""CPU-only reuse of unchanged Base evaluation; never reuses trained weights."""

import hashlib
from pathlib import Path
import shutil

from steer_opd.resumable_core import ROOT, atomic_json, digest, read_json


def main():
    cfg = read_json(ROOT / "configs/figure8_shared_top16_v2.json")
    old_run = ROOT / "runs/figure8_first_run_20261112"
    old = read_json(old_run / "config.json")
    keys = ["student", "teacher", "shared_vocabulary_size", "max_prompt_tokens", "max_response_tokens",
            "eval_samples", "eval_temperature", "eval_top_p", "eval_max_tokens", "diagnostic_questions", "seed"]
    for key in keys:
        if cfg[key] != old[key]:
            raise ValueError(f"Base reuse protocol changed: {key}")
    for name in ["dev", "aime2025"]:
        if read_json(ROOT / cfg["data_dir"] / f"{name}.json") != read_json(ROOT / old["data_dir"] / f"{name}.json"):
            raise ValueError(f"Base reuse questions/encoding changed: {name}")
    if read_json(old_run / "evaluation/base/accuracy.json")["samples"] != 30 * cfg["eval_samples"]:
        raise ValueError("Incomplete Base accuracy")
    for location, max_tokens, temperature, top_p, expected in [
        ("evaluation/base/answers", cfg["eval_max_tokens"], cfg["eval_temperature"], cfg["eval_top_p"], 480),
        ("fixed_prefixes", cfg["max_response_tokens"], 1.0, 1.0, cfg["diagnostic_questions"]),
    ]:
        manifest = read_json(old_run / location / "job.json")
        job = manifest["job"]
        for key, value in {"model": cfg["student"], "support_size": cfg["shared_vocabulary_size"],
                           "max_tokens": max_tokens, "temperature": temperature, "top_p": top_p,
                           "vector": None, "head_dtype": "float32", "normalization": "native"}.items():
            if job.get(key) != value:
                raise ValueError(f"Original Base job protocol changed: {location}/{key}")
        if manifest["hash"] != digest(job) or len(job["requests"]) != expected:
            raise ValueError("Base reuse job hash/count mismatch")
        accepted = set(manifest.get("accepted_job_hashes", []) + [manifest["hash"]])
        for request in job["requests"]:
            row = read_json(old_run / location / f"{request['key']}.json")
            if row["job_hash"] not in accepted or any(row[k] != v for k, v in request.items()):
                raise ValueError("Base response provenance mismatch")
        if not (old_run / location / "complete.json").exists():
            raise ValueError("Incomplete Base answers/prefixes")
    new_run = ROOT / cfg["run_dir"]
    files = {}
    for name in ["evaluation/base", "fixed_prefixes", "fixed_teacher_scores"]:
        for source in sorted((old_run / name).rglob("*")):
            if not source.is_file():
                continue
            relative = source.relative_to(old_run)
            target = new_run / relative
            expected = hashlib.sha256(source.read_bytes()).hexdigest()
            if target.exists():
                if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
                    raise ValueError(f"Refuse to overwrite changed reused output: {relative}")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
                raise ValueError("Base copy hash mismatch")
            files[str(relative)] = expected
    atomic_json(new_run / "initial_evaluation_reuse.json", {
        "source_run": str(old_run.relative_to(ROOT)), "source_data": old["data_dir"],
        "target_data": cfg["data_dir"], "protocol_matches": {key: cfg[key] for key in keys},
        "copied_files_sha256": files, "reused_samples": 480, "new_gpu_work": False,
        "trained_weights_reused": False, "both_groups_share_step_zero": True})
    print(f"Reused Base evaluation and fixed prefixes/scores: {len(files)} files, identical SHA256")


if __name__ == "__main__":
    main()
