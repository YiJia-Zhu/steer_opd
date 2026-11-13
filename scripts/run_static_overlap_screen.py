#!/usr/bin/env python3
"""Prepare frozen masks on CPU, then optionally score all conditions, never train.

Inputs are unsteered source rows, raw student rollout Top16, teacher caches,
explicit metric identities and four S0 banks. Native IDs require the same
mapping. The separate Compass-style adapter uses aligned byte continuations;
it never labels its result as native cross-tokenizer token-ID overlap.
"""

import argparse
import os
from pathlib import Path

import torch

from steer_opd.direction_bank import file_sha256
from steer_opd.overlap_tuning import paired_summary, position_metrics
from steer_opd.resumable_core import ROOT, atomic_json, atomic_torch, digest, read_json
from steer_opd.static_masks import freeze_static_mask, native_overlap, validate_native_score
from steer_opd.static_screen import candidate_vectors, score_native_candidate
from steer_opd.method_config import Method
from steer_opd.execution_policy import require_default_scheduler_options, require_execution_allowed


def path(value):
    result = Path(value)
    result = result if result.is_absolute() else ROOT / result
    if not result.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("Static screen inputs/outputs must stay within the project")
    return result


def prepare(job):
    from steer_opd.compass_alignment import METRIC
    if job.get("identity", {}).get("metric") == METRIC:
        from steer_opd.compass_job import prepare_screen
        return prepare_screen(job)
    if job.get("stage") != "static_overlap_screen" or job.get("optimizer_updates") != 0:
        raise ValueError("This entrypoint only accepts explicit zero-update static screens")
    if job["identity"]["metric"] != "native_overlap_at_16":
        raise ValueError("This native metric adapter cannot silently substitute for a cross-tokenizer event scorer")
    rows = read_json(path(job["rows"]))
    if not rows or len({row.get("key", row["id"]) for row in rows}) != len(rows):
        raise ValueError("Source response keys must be unique")
    masks, teachers = {}, {}
    banks = {name: torch.load(path(value), map_location="cpu", weights_only=True) for name, value in job["banks"].items()}
    conditions = job["conditions"]
    if len({c["id"] for c in conditions}) != len(conditions):
        raise ValueError("Duplicate candidate IDs")
    for condition in conditions:
        if Path(condition["id"]).name != condition["id"] or condition["id"] in (".", "..", "baseline", "masks"):
            raise ValueError("Unsafe or reserved candidate ID")
        if condition["gate"] not in ("all", "static_overlap"):
            raise ValueError("Only all/static_overlap are accepted by the static entrypoint")
        Method(direction=condition["direction"], granularity=condition["granularity"],
               layers=tuple(condition["layers"]), gate=condition["gate"], overlap_threshold=job["threshold"],
               operator=condition.get("operator", "slerp"), alpha=condition.get("alpha", .1) if condition.get("operator", "slerp") == "slerp" else None,
               rho=condition.get("rho", .03))
    for row in rows:
        key = row.get("key", row["id"])
        if Path(key).name != key or key in (".", ".."):
            raise ValueError("Response keys cannot contain path separators")
        if job.get("source_dir"):
            from steer_opd.rollout_topk import load_rollout_topk
            student_path = path(job["source_dir"]) / row["rollout_topk"]
            student = load_rollout_topk(path(job["source_dir"]), row, 16, job["support_size"])
            if row.get("model") != job["student"]:
                raise ValueError("Rollout model differs from frozen scoring student")
        else:
            student_path = path(job["student_scores"]) / f"{key}.pt"
            student = torch.load(student_path, map_location="cpu", weights_only=True)
        teacher_path = path(job["teacher_scores"]) / f"{key}.pt"
        teachers[key] = torch.load(teacher_path, map_location="cpu", weights_only=True)
        if student.get("source") not in ("vllm_rollout_raw_logprobs", "hf_controlled_rollout_raw_logprobs", "hf_unsteered_rollout_raw_logprobs"):
            raise ValueError("Source Top16 must come from an unsteered rollout, not a controlled probe")
        source_hash = digest({"prompt_token_ids": row["prompt_token_ids"], "token_ids": row["token_ids"]})
        for score in ([teachers[key]] if job.get("source_dir") else [student, teachers[key]]):
            if score.get("source_token_hash") != source_hash or score.get("support_size") != job["support_size"]:
                raise ValueError("Probability cache prefix or output support mismatch")
        if teachers[key].get("teacher_checkpoint") != job["identity"]["teacher_checkpoint"]:
            raise ValueError("Teacher probability cache checkpoint mismatch")
        for score in (student, teachers[key]):
            validate_native_score(score, len(row["token_ids"]), job["support_size"])
        identity = {**job["identity"], "student_score_sha256": file_sha256(student_path),
                    "teacher_score_sha256": file_sha256(teacher_path)}
        if identity["student_checkpoint"] != job["student"]:
            raise ValueError("Source checkpoint differs from scoring student")
        overlap = native_overlap(student["ids"], teachers[key]["ids"], identity)
        masks[key] = freeze_static_mask(row, overlap, job["threshold"], identity)
        for condition in conditions:
            bank = banks[condition["bank"]]
            if row["id"] not in bank["valid_ids"] or bank["identity"]["checkpoint"] != job["student"]:
                raise ValueError("Every scheme must share the same usable questions and S0 checkpoint")
            candidate_vectors(bank, row, condition)
    output = path(job["output"])
    frozen = {"job_hash": digest(job), "source_rows_hash": digest(rows),
              "masks": {k: v["mask_id"] for k, v in masks.items()},
              "bank_hashes": {k: file_sha256(path(p)) for k, p in job["banks"].items()}, "parameter_updates": 0}
    if (output / "manifest.json").exists() and read_json(output / "manifest.json") != frozen:
        raise ValueError("Cannot resume changed static screen inputs")
    for key, artifact in masks.items():
        atomic_json(output / "masks" / f"{key}.json", artifact)
    atomic_json(output / "manifest.json", frozen)
    return rows, masks, teachers, banks


def run(job, rows, masks, teachers, banks):
    require_execution_allowed(job)
    require_default_scheduler_options(job)
    from steer_opd.resumable_core import load_model, require_gpu
    from steer_opd.static_plotting import plot_comparisons
    from steer_opd.compass_job import model_path
    os.environ["STEER_OPD_SUPPORT_SIZE"] = str(job["support_size"])
    output = path(job["output"])
    metric_adapter = getattr(teachers, "metric_adapter", None)
    plot_keys = ([r.get("key", r["id"]) for r in read_json(path(job["rows"]))[:3]]
                 if metric_adapter is not None else None)
    phase_binding = digest(read_json(output / "manifest.json"))
    def cached(name):
        marker = output / name / "complete.json"
        if not marker.exists():
            return None
        commit = read_json(marker)
        if commit["binding"] != phase_binding:
            raise ValueError("Cached static phase has a different source/config identity")
        records = {}
        for key, sha in commit["files"].items():
            target = output / name / f"{key}.pt"
            if not target.exists() or file_sha256(target) != sha:
                raise ValueError("Committed static phase record missing or changed")
            records[key] = torch.load(target, map_location="cpu", weights_only=True)
        if set(records) != {r.get("key", r["id"]) for r in rows}:
            raise ValueError("Committed static phase is missing source rows")
        return records
    def commit(name, records):
        for key, record in records.items():
            atomic_torch(output / name / f"{key}.pt", record)
        atomic_json(output / name / "complete.json", {"binding": phase_binding, "parameter_updates": 0,
                    "files": {key: file_sha256(output / name / f"{key}.pt") for key in records}})
    required = ["baseline", *[c["id"] for c in job["conditions"]]]
    ready = {name: cached(name) for name in required}
    if all(ready[name] is not None for name in required) and (output / "complete.json").exists():
        # Rebuild requested CPU plots from cached data if necessary.
        for condition in job["conditions"]:
            plot_comparisons(rows, ready["baseline"], ready[condition["id"]], output / condition["id"] / "comparison", requested_keys=plot_keys)
        print("All static conditions already complete; reused committed probability records")
        return
    needs_model = any(ready[name] is None for name in required)
    backend = job.get("backend", "hf_batched_static")
    if backend not in ("hf_batched_static", "easysteer"):
        raise ValueError("Unknown static backend; no automatic backend substitution")
    if needs_model:
        if backend == "easysteer":
            from steer_opd.easysteer_static import require_easysteer_gpu
            require_easysteer_gpu()
        else:
            require_gpu()
    model = None
    if needs_model and backend == "hf_batched_static":
        if metric_adapter is None:
            model = load_model(model_path(job["student"]))
        else:
            from steer_opd.compass_scoring import load_text_scorer
            model = load_text_scorer(model_path(job["student"]), metric_adapter.student_codec)
    elif needs_model:
        from steer_opd.easysteer_static import engine_options, require_easysteer
        require_easysteer()
        if metric_adapter is not None and metric_adapter.student_codec.valid_ids != list(range(job["support_size"])):
            raise ValueError("EasySteer support cropping requires a contiguous defined student vocabulary")
        os.environ["STEER_OPD_STATIC_SUPPORT"] = str(job["support_size"])
        from vllm import LLM
        model = LLM(model=str(model_path(job["student"])), dtype="bfloat16", tensor_parallel_size=1,
            max_model_len=max(len(r["prompt_token_ids"])+len(r["token_ids"]) for r in rows),
            gpu_memory_utilization=job.get("gpu_memory_utilization", .75),
            logprobs_mode="raw_logprobs", max_logprobs=16, generation_config="vllm",
            worker_cls="steer_opd.easysteer_worker.StaticWorker", **engine_options())
    def evaluate(condition):
        bank = banks[condition["bank"]] if condition else next(iter(banks.values()))
        if backend == "hf_batched_static":
            return score_native_candidate(model, rows, teachers, masks, bank, condition,
                baseline=baseline if condition else None, batch_size=job.get("batch_size", 4),
                chunk=job.get("chunk", 128), metric_adapter=metric_adapter)
        from steer_opd.easysteer_static import build_spec, fixed_prefix_scores
        specs, meta = [], []
        for row in rows:
            key = row.get("key", row["id"])
            if condition:
                vectors, vector_id = candidate_vectors(bank, row, condition)
                requested = masks[key]["requested_mask"] if condition["gate"] == "static_overlap" else [True]*len(row["token_ids"])
                nonzero = any(bool(torch.count_nonzero(v)) for v in vectors.values())
                active = nonzero and condition.get("alpha", .1) != 0
                actual = [g and active for g in requested]
                spec = build_spec(row, actual, vectors, operator=condition.get("operator", "slerp"), alpha=condition.get("alpha", .1))
            else:
                vector_id, requested, actual, spec, nonzero = None, [False]*len(row["token_ids"]), [False]*len(row["token_ids"]), False, False
            specs.append(spec)
            meta.append({"mask_id": masks[key]["mask_id"], "source_binding": masks[key]["source_binding"],
                "binding": digest({"mask_id": masks[key]["mask_id"], "condition": condition, "vector_id": vector_id}),
                "vector_id": vector_id, "condition": condition, "requested_mask": torch.tensor(requested),
                "steering_mask": torch.tensor(actual), "relative_perturbation_available": False,
                "zero_vector_no_intervention": bool(condition is not None and not nonzero)})
        scores = fixed_prefix_scores(model, rows, specs, include_sampled=metric_adapter is not None)
        return {row.get("key", row["id"]): {**s,
                    **(position_metrics(s, teachers[row.get("key", row["id"])]) if metric_adapter is None else
                       metric_adapter(row.get("key", row["id"]), s, teachers[row.get("key", row["id"])])), **m}
                for row, s, m in zip(rows, scores, meta)}
    baseline = ready["baseline"]
    if baseline is None:
        baseline = evaluate(None)
        for row in rows:
            key = row.get("key", row["id"])
            baseline[key]["source_rollout_vs_replay_delta"] = baseline[key]["overlap"].double()-torch.tensor(
                [float("nan") if v is None else v for v in masks[key]["source_overlap"]], dtype=torch.float64)
        commit("baseline", baseline)
    summaries = {}
    for condition in job["conditions"]:
        records = ready[condition["id"]]
        if records is None:
            records = evaluate(condition)
            commit(condition["id"], records)
        # Aggregate multiple responses by question before paired_summary.
        def by_question(values):
            groups = {}
            for row in rows:
                groups.setdefault(row["id"], []).append(values[row.get("key", row["id"])])
            return {qid: {name: torch.cat([v[name][v.get("alignment_mask", torch.ones_like(v[name], dtype=torch.bool))] for v in items]) for name in
                          ("overlap", "student_shared_mass", "teacher_shared_mass")} for qid, items in groups.items()}
        summaries[condition["id"]] = {"condition": condition, "metric": job["identity"]["metric"],
                                      **paired_summary(by_question(records), by_question(baseline))}
        atomic_json(output / "summaries.json", summaries)
        plot_comparisons(rows, baseline, records, output / condition["id"] / "comparison", requested_keys=plot_keys)
    atomic_json(output / "complete.json", {"status": "awaiting_user_review", "parameter_updates": 0,
                "candidates": len(summaries), "backend": backend, "automatic_opd": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True)
    parser.add_argument("--execute", action="store_true", help="Score on an admitted GPU; default only prepares masks on CPU")
    args = parser.parse_args()
    job = read_json(args.job)
    require_default_scheduler_options(job)
    if args.execute:
        require_execution_allowed(job)
    prepared = prepare(job)
    if args.execute:
        run(job, *prepared)
    else:
        print(f"CPU preparation complete: {len(prepared[0])} sources, {len(job['conditions'])} candidates; no model loaded")


if __name__ == "__main__":
    main()
