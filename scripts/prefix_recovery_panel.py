#!/usr/bin/env python3
"""Small fixed-Base-prefix continuation diagnostic; never a training gate."""

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from transformers import AutoTokenizer

from figure8_report import grade_one
from steer_opd.resumable_core import ROOT, atomic_json, atomic_torch, digest, read_json


def execute(driver):
    root = driver.run / "prefix_recovery"
    if (root / "results.json").exists():
        return
    cfg = read_json(ROOT / "configs/full_scheme_precheck_20261112.json")
    precheck = ROOT / cfg["run_dir"]
    rows = read_json(precheck / "select_rows.json") + read_json(precheck / "confirm_rows.json")
    tokenizer = AutoTokenizer.from_pretrained(ROOT / cfg["student"], local_files_only=True)
    selection_path = root / "prefixes.json"
    if selection_path.exists():
        prefixes = read_json(selection_path)
    else:
        rng = np.random.default_rng(42)
        candidates = [[] for _ in range(4)]
        for row in rows:
            phase = "select" if any(row["id"] == value["id"] for value in read_json(precheck / "select_rows.json")) else "confirm"
            record = torch.load(precheck / phase / "baseline" / f"{row['id']}.pt", weights_only=True)
            for position, overlap in enumerate(record["overlap"].tolist()):
                candidates[min(3, int(overlap*4))].append((row, position, overlap))
        prefixes = []
        for bucket, values in enumerate(candidates):
            chosen_questions = set()
            for index in rng.permutation(len(values)):
                row, position, overlap = values[index]
                if row["id"] in chosen_questions:
                    continue
                prefix_text = tokenizer.decode(row["token_ids"][:position], skip_special_tokens=True)
                if "\\boxed" in prefix_text:
                    continue  # Already answered prefixes confound recovery.
                prefixes.append({"key": f"bucket{bucket}_{len(chosen_questions)}", "id": row["id"], "answer": row["answer"],
                    "prompt_token_ids": row["prompt_token_ids"]+row["token_ids"][:position], "seed": 700000+len(prefixes),
                    "base_prefix_text": prefix_text, "overlap_before": overlap, "position": position, "bucket": bucket,
                    "uses_future_for_gate": False, "diagnostic_selection_only": True})
                chosen_questions.add(row["id"])
                if len(chosen_questions) == 2:
                    break
        if not prefixes:
            raise ValueError("No eligible fixed prefixes for continuation analysis")
        atomic_json(selection_path, prefixes)
    models = {"teacher": ROOT / cfg["teacher"], "base": ROOT / cfg["student"],
        "vanilla50": driver.run / "vanilla/seed42/vanilla/checkpoints/step_0050/model",
        "main50": driver.run / "main_222/seed42/steer/checkpoints/step_0050/model"}
    # Additional inference-only anchor: the original global T-S direction,
    # starting at the cut (earlier prefix states receive no steering).
    selected = read_json(driver.run / "selected_intervention.json")["condition"]
    bank = torch.load(precheck / "directions/initial/vectors.pt", weights_only=True)
    bank["layers"] = {str(layer): bank["layers"][str(layer)] for layer in selected["layers"]}
    bank["method"].update(layers=selected["layers"], rho=selected["rho"], gate="all")
    bank["bank_id"] = digest({"parent": bank["bank_id"], "condition": selected, "purpose": "prefix_TS_anchor"})
    vector = root / "TS_global_from_cut.pt"
    atomic_torch(vector, bank)
    models["base_TS_from_cut"] = ROOT / cfg["student"]
    results = {}
    for role, model in models.items():
        output = root / role
        driver.generate(f"recovery_{role}", model, prefixes, output / "answers", cfg["max_response_tokens"], 1., 1.,
                        vector if role == "base_TS_from_cut" else None, logprobs=None)
        grades = output / "grades"
        pending = []
        for request in prefixes:
            row = read_json(output / "answers" / f"{request['key']}.json")
            if not (grades / f"{request['key']}.json").exists():
                pending.append((request["key"], {**row, "text": request["base_prefix_text"]+row["text"]}, str(grades)))
        if pending:
            with ProcessPoolExecutor(max_workers=4) as pool:
                list(pool.map(grade_one, pending))
        values = [read_json(grades / f"{request['key']}.json") for request in prefixes]
        if any(value["grading_status"] == "exception" for value in values):
            raise RuntimeError("Continuation grading exceptions need review")
        results[role] = {"correct": sum(value["correct"] for value in values), "prefixes": len(values),
            "by_overlap_bucket": {str(bucket): {"n": sum(request["bucket"] == bucket for request in prefixes),
                "correct": sum(value["correct"] for request, value in zip(prefixes, values) if request["bucket"] == bucket)}
                for bucket in range(4)}, "steering": role == "base_TS_from_cut"}
    atomic_json(root / "results.json", {"models": results, "selection": "frozen ordinary Base dev prefixes, two questions per overlap bucket",
        "teacher_reads_student_prefix": True, "used_for_training_or_threshold_selection": False,
        "TS_anchor_is_not_online_TR_gate": True, "small_diagnostic_not_significance_evidence": True})
