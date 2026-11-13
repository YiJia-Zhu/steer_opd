#!/usr/bin/env python3
"""Durable current-pair panel supervisor. GPU admission is always per child.

No claim that queued/configured jobs are completed experiments. Existing runs
are preserved and all variants begin from the original Base.
"""

import argparse
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from transformers import AutoTokenizer

from run_figure8_campaign import Campaign, relative
from steer_opd.full_experiment import panel, gpu_seconds, paired_accuracy_interval
from steer_opd.gate_controls import calibrate_threshold
from steer_opd.math_data import prompt_ids
from steer_opd.resumable_core import ROOT, atomic_json, digest, read_json


RUN = ROOT / "runs/full_scheme_20261112"
PRECHECK = ROOT / "configs/full_scheme_precheck_20261112.json"


def immutable_json(path, value):
    if path.exists() and read_json(path) != value:
        raise ValueError(f"Frozen panel identity changed: {path}")
    atomic_json(path, value)


def copy_cache(source, target):
    if not target.exists():
        shutil.copytree(source, target, copy_function=os.link)


def wait_precheck():
    while True:
        path = RUN / "precheck/campaign.json"
        state = read_json(path) if path.exists() else {}
        if state.get("status") == "awaiting_user_review":
            # User has now explicitly authorized the full panel, superseding
            # the old stop point. The diagnostic entry itself stays unchanged.
            return
        if state.get("status") == "failed_resumable":
            raise RuntimeError("Precheck failed resumably; inspect its actual stage logs")
        record_path = RUN / "precheck/supervisor.json"
        pid = read_json(record_path).get("pid") if record_path.exists() else None
        cmdline = Path(f"/proc/{pid}/cmdline")
        alive = cmdline.exists() and b"run_pre_opd_overlap_tuning.py" in cmdline.read_bytes()
        if not alive:
            subprocess.run([sys.executable, "scripts/run_pre_opd_overlap_tuning.py", "--config", str(PRECHECK), "--execute"],
                           cwd=ROOT, check=True)
            continue
        atomic_json(RUN / "panel_status.json", {"status": "waiting_precheck", "updated_at": time.time(),
            "precheck_phase": state.get("phase"), "precheck_pid": pid, "gpu_limit": 1})
        time.sleep(10)


def select_and_calibrate(driver):
    precheck = RUN / "precheck"
    summaries = read_json(precheck / "select/summaries.json")
    single = [row for row in summaries.values() if row["condition"].get("control") == "positive"]
    single.sort(key=lambda row: (-row["question_weighted_delta_pp"], row["condition"]["id"]))
    first = single[0]["condition"]["layers"][0]
    second = next(row["condition"]["layers"][0] for row in single if row["condition"]["layers"][0] != first)
    cfg = read_json(PRECHECK)
    rows = read_json(precheck / "select_rows.json")
    conditions = [{"id": f"L{first}_{second}_rho{rho}", "kind": "bank", "layers": [first, second],
                   "rho": rho, "control": "positive", "seed": 42} for rho in cfg["sweep_rhos"]]
    bank = relative(precheck / "directions/initial/vectors.pt")
    def score(phase, rows_path, candidates):
        job = {"config": relative(PRECHECK), "bank": bank, "rows": relative(rows_path),
            "teacher_scores": cfg["fixed_prefix_source"] + "/fixed_teacher_scores",
            "output": relative(RUN / phase), "conditions": candidates}
        path = RUN / "jobs" / f"{phase}.json"
        immutable_json(path, job)
        if not (RUN / phase / "complete.json").exists():
            driver.stage(phase, ["scripts/pre_opd_overlap_stage.py", "--job", str(path)])
        return read_json(RUN / phase / "summaries.json")
    multi = score("multilayer_select", precheck / "select_rows.json", conditions)
    choices = [*single, *multi.values()]
    choices.sort(key=lambda row: (-row["question_weighted_delta_pp"], row["condition"]["id"]))
    selected = choices[0]["condition"]
    score("multilayer_confirm", precheck / "confirm_rows.json", [selected])
    immutable_json(RUN / "zero_update_selection.json", {"condition": selected,
        "selection_source": "dev-select only; confirm never used for choosing", "updates": 0,
        "does_not_establish_learning": True})
    overlaps, gaps, lengths = [], [], []
    for row in rows:
        record = torch.load(precheck / "select/baseline" / f"{row['id']}.pt", weights_only=True)
        overlaps.extend(record["overlap"].tolist())
        gaps.extend(record["signed_entropy_gap"].tolist())
        lengths.append(len(record["overlap"]))
    calibration = {"overlap": calibrate_threshold(overlaps, .2, "lt"),
        "entropy": calibrate_threshold(gaps, .2, "gt", lower_bound=0.),
        "overlap_candidates": [calibrate_threshold(overlaps, rate, "lt") for rate in [.1, .2, .3]],
        "late_start_token": int(np.median(lengths)/2), "late_boundary_rule": "half median completed Base-select length, frozen before training", "uses_test": False,
        "source_ids": [row["id"] for row in rows], "source": "frozen ordinary Base select prefixes",
        "online_realized_rate_can_change": True}
    immutable_json(RUN / "gate_calibration.json", calibration)
    return selected, calibration


def learning_selection(driver, base, zero_condition, calibration):
    """Equal small search grids; choose by actual no-hook dev learning."""
    single = read_json(RUN / "precheck/select/summaries.json")
    anchor = next(value["condition"] for value in single.values()
                  if value["condition"].get("layers") == [13] and value["condition"].get("rho") == .03)
    multi = max(read_json(RUN / "multilayer_select/summaries.json").values(), key=lambda value: value["question_weighted_delta_pp"])["condition"]
    conditions = list({condition["id"]: condition for condition in [zero_condition, anchor, multi]}.values())
    immutable_json(RUN / "learning_selection_candidates.json", conditions)
    baseline_output = RUN / "learning_selection/base_dev"
    dev = read_json(ROOT / base["data_dir"] / "dev.json")[base["diagnostic_questions"]:]
    requests = [{"key": f"{index:04d}_00", "id": row["id"], "answer": row["answer"],
                 "prompt_token_ids": row["student_prompt_ids"], "seed": 500000+index} for index, row in enumerate(dev)]
    driver.generate("eval_learning_selection_base", ROOT / base["student"], requests, baseline_output / "answers",
        base["max_response_tokens"], base["eval_temperature"], base["eval_top_p"], logprobs=None)
    if not (baseline_output / "accuracy.json").exists():
        driver.cpu_stage("grade_learning_selection_base", ["scripts/figure8_report.py", "--grade", str(baseline_output), "--grade-only"])
    baseline_accuracy = read_json(baseline_output / "accuracy.json")["accuracy"]
    trials, chosen = [], {}
    for direction in ("T-R", "T-S", "R-S"):
        for condition in conditions:
            prototype = next(cfg for name, seed, cfg in panel(base, condition, calibration) if name == "main_222" and seed == 42)
            name = f"{direction.replace('-', '')}_{condition['id'].replace('.', 'p')}".lower()
            cfg = {**prototype, "id": f"selection_{name}", "steps": 10, "eval_steps": [], "dev_eval_steps": [10],
                "additional_evaluation_sets": [], "direction": direction,
                "run_dir": f"runs/full_scheme_20261112/learning_selection/{name}",
                "data_dir": f"data/full_scheme_20261112/learning_selection/{name}",
                "steering_method": {**prototype["steering_method"], "direction": direction}}
            cfg["deviations"] = ["10-update training-side hyperparameter selection; fresh Base; no test selection",
                                  "same candidate grid and dev96 for all3 directions; independent final runs start fresh"]
            path = RUN / "configs" / f"selection_{name}.json"
            immutable_json(path, cfg)
            seed_existing_evaluation(cfg)
            run = ROOT / cfg["run_dir"]
            metric = run / "evaluation/selection_dev_steer_0010/accuracy.json"
            state_path = run / "campaign.json"
            complete = state_path.exists() and read_json(state_path).get("status") == "complete"
            if not metric.exists() or not complete:
                driver.cpu_stage(f"learning_selection_{name}", ["scripts/run_figure8_campaign.py", "--config", str(path)], required=False)
                failed = read_json(state_path) if state_path.exists() else {}
                if failed.get("status") != "complete":
                    if "BootstrapQualityError" in failed.get("error", ""):
                        trials.append({"direction": direction, "condition": condition, "status": "bootstrap_quality_budget_exhausted",
                                       "run_dir": cfg["run_dir"], "training_gpu_seconds": gpu_seconds(run)})
                        continue
                    raise RuntimeError(f"Selection trial failed: {run}; inspect its actual failure")
            value = read_json(metric)
            nohook = read_json(run / "evaluation/steer_0010/overlap.json")
            trial = {"direction": direction, "condition": condition, "accuracy": value["accuracy"],
                "delta_accuracy": value["accuracy"]-baseline_accuracy, "nohook_overlap16": nohook["overlap16"],
                "training_gpu_seconds": gpu_seconds(run), "run_dir": cfg["run_dir"]}
            trials.append(trial)
            immutable_json(run / "selection_result.json", trial)
            # This10-step trial is COMPLETE; the final50-step run starts fresh.
            # Keep weights, directions and all raw trajectories; only its now
            # unused optimizer is released, never an unfinished run's state.
            completed = read_json(run / "campaign.json")
            resume = run / "steer/checkpoints/step_0010/resume.pt"
            if completed.get("status") != "complete" or not list((resume.parent / "model").glob("*.safetensors")):
                raise ValueError("Cannot finalize an incomplete selection trial")
            if resume.exists():
                resume.unlink()
            atomic_json(run / "trial_finalized.json", {"updates": 10, "all_requested_work_complete": True,
                "optimizer_retained": False, "weights_and_raw_evidence_retained": True,
                "pipeline_resume_does_not_need_more_trial_updates": True})
        candidates = [value for value in trials if value["direction"] == direction and "accuracy" in value]
        if not candidates:
            if direction == "T-R":
                raise RuntimeError("Main direction has no completed training-side selection")
            continue
        best = min(candidates, key=lambda value: (-value["accuracy"], -value["nohook_overlap16"],
                    value["training_gpu_seconds"], value["condition"]["id"]))
        chosen[direction] = best["condition"]
    immutable_json(RUN / "learner_selection.json", {"trials": trials, "chosen": chosen,
        "baseline_dev_accuracy": baseline_accuracy, "selection": "nohook dev96 accuracy; overlap/cost are predeclared tie breaks",
        "uses_test": False, "maximum_selection_updates": len(trials)*10})
    immutable_json(RUN / "selected_intervention.json", {"condition": chosen["T-R"],
        "selection_source": "actual10-update nohook dev selection on equal direction grids", "final_run_starts_fresh_Base": True})
    return chosen["T-R"], chosen


def references(driver):
    raw = RUN / "references/raw"
    for role in ("teacher", "student"):
        copy_cache(RUN / f"precheck/calibration/{role}", raw / role)
    rows = read_json(ROOT / driver.cfg["data_dir"] / "direction.json")
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "assets/models/Qwen3-0.6B", local_files_only=True)
    requests = [{"key": row["id"], "id": row["id"], "answer": row["answer"],
                 "prompt_token_ids": prompt_ids(tokenizer, row["question"], enable_thinking=False), "seed": 42+index}
                for index, row in enumerate(rows)]
    driver.generate("calibration_reference", ROOT / "assets/models/Qwen3-0.6B", requests, raw / "reference",
        driver.cfg["calibration_max_tokens"], driver.cfg["calibration_temperature"], driver.cfg["calibration_top_p"], logprobs=None)
    for name, changes in [("common", {}), ("template_length", {"normalize_template": True, "match_length": True}),
                          ("both_correct", {"both_correct": True})]:
        output = RUN / "references" / name
        if (output / "manifest.json").exists():
            continue
        job = {"student": driver.cfg["student"], "questions": driver.cfg["data_dir"] + "/direction.json",
            "roles": {role: relative(raw / role) for role in ("teacher", "reference", "student")},
            "output": relative(output), "minimum_pairs": 1, **changes}
        path = RUN / "jobs" / f"references_{name}.json"
        immutable_json(path, job)
        driver.cpu_stage(f"references_{name}", ["scripts/prepare_reference_controls.py", "--job", str(path)], required=name == "common")
    independent = RUN / "references/independent_raw"
    independent_rows = read_json(ROOT / "data/figure8_first_run_20261112/direction.json")[:100]
    main_ids = {row["id"] for row in rows}
    if main_ids & {row["id"] for row in independent_rows}:
        raise ValueError("Independent direction questions overlap OPD")
    questions_path = RUN / "references/independent_questions.json"
    immutable_json(questions_path, independent_rows)
    for role in ("teacher", "student"):
        copy_cache(ROOT / f"runs/figure8_first_run_20261112/calibration/{role}", independent / role)
    requests = [{"key": row["id"], "id": row["id"], "answer": row["answer"],
                 "prompt_token_ids": prompt_ids(tokenizer, row["question"], enable_thinking=False), "seed": 42+index}
                for index, row in enumerate(independent_rows)]
    driver.generate("independent_calibration_reference", ROOT / "assets/models/Qwen3-0.6B", requests,
        independent / "reference", driver.cfg["calibration_max_tokens"], driver.cfg["calibration_temperature"],
        driver.cfg["calibration_top_p"], logprobs=None)
    output = RUN / "references/independent_common"
    if not (output / "manifest.json").exists():
        job = {"student": driver.cfg["student"], "questions": relative(questions_path),
               "roles": {role: relative(independent / role) for role in ("teacher", "reference", "student")},
               "output": relative(output), "minimum_pairs": 8}
        path = RUN / "jobs/references_independent_common.json"
        immutable_json(path, job)
        driver.cpu_stage("independent_references_common", ["scripts/prepare_reference_controls.py", "--job", str(path)])


def seed_existing_evaluation(cfg):
    source = ROOT / cfg["evaluation_cache_source"]
    original = read_json(ROOT / "configs/figure8_shared_top16_v2.json")
    keys = ("student", "teacher", "shared_vocabulary_size", "eval_samples", "eval_temperature", "eval_top_p", "eval_max_tokens")
    if any(cfg[key] != original[key] for key in keys):
        raise ValueError("Cannot reuse Base evaluation under changed scientific protocol")
    run = ROOT / cfg["run_dir"]
    for name in ("fixed_prefixes", "fixed_teacher_scores", "evaluation/base"):
        if (source / name).exists():
            copy_cache(source / name, run / name)
    atomic_json(run / "initial_evaluation_reuse.json", {"source": cfg["evaluation_cache_source"],
        "protocol_matches": {key: cfg[key] for key in keys}, "shared_base_for_all_groups": True})


def report(manifest):
    rows = []
    for item in manifest:
        cfg, name, seed = item["config"], item["name"], item["seed"]
        run = ROOT / cfg["run_dir"]
        group = cfg["groups"][0]
        state = read_json(run / "campaign.json") if (run / "campaign.json").exists() else {}
        endpoint = state.get("steps_per_group", cfg["steps"])
        path = run / f"evaluation/{group}_{endpoint:04d}/accuracy.json"
        rows.append({"name": name, "seed": seed, "status": state.get("status", "queued"),
            "accuracy": read_json(path) if path.exists() else None, "training_gpu_seconds": gpu_seconds(run),
            "dev_accuracy_by_step": {p.parent.name: read_json(p) for p in (run / "evaluation").glob("selection_dev_*/accuracy.json")},
            "run_dir": cfg["run_dir"]})
    summary = {}
    for name in {row["name"] for row in rows}:
        observed = [row["accuracy"]["accuracy"] for row in rows if row["name"] == name and row["accuracy"]]
        if observed:
            summary[name] = {"completed_seeds": len(observed), "mean_accuracy": float(np.mean(observed)),
                             "sample_std": float(np.std(observed, ddof=1)) if len(observed)>1 else None}
    pairs = []
    for seed in (42, 43, 44):
        main = next((r for r in rows if r["name"] == "main_222" and r["seed"] == seed), None)
        base = next((r for r in rows if r["name"] == "vanilla" and r["seed"] == seed), None)
        if main and base and main["accuracy"] and base["accuracy"]:
            pairs.append({"seed": seed, **paired_accuracy_interval(main["accuracy"]["per_question"], base["accuracy"]["per_question"])})
    atomic_json(RUN / "panel_results.json", {"experiments": rows, "seed_summary": summary, "paired_main_vs_vanilla": pairs,
        "gpu_acceptance_evidence": "per-run acceptance files; scheduling is not acceptance"})


def execute():
    subprocess.run([sys.executable, "scripts/freeze_full_inputs.py"], cwd=ROOT, check=True)
    wait_precheck()
    base = read_json(ROOT / "configs/figure8_shared_top16_v2.json")
    config = {**base, "id": "full_scheme_driver_20261112", "authorized": True, "execution_hold": None,
              "run_dir": relative(RUN), "reference": "assets/models/Qwen3-0.6B"}
    path = ROOT / "configs/full_scheme_driver_20261112.json"
    immutable_json(path, config)
    driver = Campaign(path)
    driver.resume_command = ".venv-opd/bin/python scripts/run_full_scheme.py --execute"
    try:
        zero_selected, calibration = select_and_calibrate(driver)
        references(driver)
        if not (RUN / "references/initial_pools/manifest.json").exists():
            driver.stage("initial_triplet_pools", ["scripts/prepare_initial_triplets.py", "--config", str(path)])
        selected, per_direction_selection = learning_selection(driver, base, zero_selected, calibration)
        selection_record = read_json(RUN / "learner_selection.json")
        preparation = {
            "zero_update_precheck_gpu_seconds": gpu_seconds(RUN / "precheck", training_only=False),
            "shared_needed_gpu_seconds": gpu_seconds(RUN, training_only=False,
                phase_prefixes=("multilayer_", "calibration_reference_", "initial_triplet_", "eval_learning_selection_base_")),
            "TR_learning_selection_gpu_seconds": sum(gpu_seconds(ROOT / value["run_dir"], training_only=False)
                for value in selection_record["trials"] if value["direction"] == "T-R")}
        preparation["additional_main_gpu_seconds"] = sum(preparation.values())
        preparation["scope"] = "zero-update discovery/calibration, needed shared references/pools, T-R model selection; excludes unrelated controls and final tests"
        immutable_json(RUN / "preparation_costs.json", preparation)
        manifest = []
        for name, seed, cfg in panel(base, selected, calibration, per_direction_selection=per_direction_selection):
            config_path = RUN / "configs" / f"{name}_seed{seed}.json"
            immutable_json(config_path, cfg)
            manifest.append({"name": name, "seed": seed, "path": relative(config_path), "config": cfg})
        immutable_json(RUN / "panel_manifest.json", manifest)
        report(manifest)
        for item in manifest[:2]:
            cfg = item["config"]
            seed_existing_evaluation(cfg)
            driver.cpu_stage(f"first_pair_10_steps_{item['name']}", ["scripts/run_figure8_campaign.py", "--config",
                str(ROOT / item["path"]), "--until-step", "10"])
            report(manifest)
        for item in manifest:
            cfg, name, seed = item["config"], item["name"], item["seed"]
            method = cfg["steering_method"]
            if (method["extractor"] == "current" and method["direction"] not in per_direction_selection
                    and cfg.get("student_text_refresh", "current") == "current"):
                atomic_json(ROOT / cfg["run_dir"] / "campaign.json", {"status": "bootstrap_quality_budget_exhausted_in_selection",
                    "not_an_efficacy_result": True, "native_student_text_not_replaced_with_S0": True})
                report(manifest)
                continue
            source = ROOT / cfg["reference_pool_dir"] / "manifest.json"
            if not source.exists() or len(read_json(source)["valid_ids"]) < cfg["minimum_direction_pairs"]:
                atomic_json(ROOT / cfg["run_dir"] / "campaign.json", {"status": "insufficient_reference_data",
                    "source": relative(source), "not_an_experimental_result": True})
                report(manifest)
                continue
            seed_existing_evaluation(cfg)
            while shutil.disk_usage(ROOT).free < 40 * 1024**3:
                atomic_json(RUN / "panel_status.json", {"status": "waiting_project_quota_space",
                    "available_bytes": shutil.disk_usage(ROOT).free, "minimum_bytes": 40*1024**3,
                    "next_experiment": name, "max_gpus": 1, "updated_at": time.time()})
                time.sleep(10)
            atomic_json(RUN / "panel_status.json", {"status": "running_panel", "name": name, "seed": seed,
                "updated_at": time.time(), "config": item["path"], "max_gpus": 1})
            driver.cpu_stage(f"experiment_{name}_seed{seed}", ["scripts/run_figure8_campaign.py", "--config", str(ROOT / item["path"])], required=False)
            state = read_json(ROOT / cfg["run_dir"] / "campaign.json")
            if state.get("status") not in ("complete", "maximum_steps_reached_compute_unmatched"):
                if "BootstrapQualityError" in state.get("error", ""):
                    state.update(status="bootstrap_quality_budget_exhausted", not_an_efficacy_result=True)
                    atomic_json(ROOT / cfg["run_dir"] / "campaign.json", state)
                    report(manifest)
                    continue
                raise RuntimeError(f"Experiment failed unexpectedly: {cfg['run_dir']}; artifacts retained")
            driver.cpu_stage(f"analysis_{name}_seed{seed}", ["scripts/analyze_full_scheme.py", "--config", str(ROOT / item["path"])])
            report(manifest)
            if name == "main_222" and seed == 42:
                from prefix_recovery_panel import execute as recovery
                recovery(driver)
        atomic_json(RUN / "panel_status.json", {"status": "panel_finished", "updated_at": time.time(),
            "results": relative(RUN / "panel_results.json"), "extensions": "SFT/cross-tokenizer pending scope clarification"})
    except BaseException as exc:
        atomic_json(RUN / "panel_status.json", {"status": "failed_resumable", "error": repr(exc), "updated_at": time.time()})
        raise
    finally:
        driver.lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute:
        RUN.mkdir(parents=True, exist_ok=True)
        with (RUN / "panel.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            execute()
    else:
        print("CPU preview: precheck → multilayer/thresholds → common references → current-pair panel. Use --execute after authorization.")
