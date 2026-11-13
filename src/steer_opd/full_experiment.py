"""Explicit experiment panel; no Cartesian explosion or hidden method changes."""

import copy
from datetime import datetime
from pathlib import Path

import numpy as np

from steer_opd.method_config import Method
from steer_opd.resumable_core import read_json


def panel(base, selection, calibration, root="runs/full_scheme_20261112", per_direction_selection=None):
    method = Method(direction="T-R", granularity="per_question", extractor="current",
        layers=tuple(selection["layers"]), rho=selection["rho"], gate="overlap",
        overlap_threshold=calibration["overlap"]["threshold"]).to_dict()
    definitions = [
        ("vanilla", {}, {"groups": ["vanilla"]}),
        ("main_222", {}, {}),
        ("direction_TS", {"direction": "T-S"}, {}),
        ("direction_RS", {"direction": "R-S"}, {}),
        ("direction_TS_fixed_self_text", {"direction": "T-S"}, {"student_text_refresh": "initial"}),
        ("direction_RS_fixed_self_text", {"direction": "R-S"}, {"student_text_refresh": "initial"}),
        ("all_positions", {"gate": "all"}, {}),
        ("ts_global_all", {"direction": "T-S", "gate": "all", "granularity": "global", "extractor": "initial"}, {}),
        ("ts_global_all_vllm", {"direction": "T-S", "gate": "all", "granularity": "global", "extractor": "initial"},
            {"generation_backend": "vllm"}),
        ("vanilla_vllm", {"gate": "all"}, {"groups": ["vanilla"], "generation_backend": "vllm"}),
        ("all_positions_vllm", {"gate": "all"}, {"generation_backend": "vllm"}),
        ("entropy_gate", {"gate": "entropy", "entropy_threshold": calibration["entropy"]["threshold"]}, {}),
        ("global_same_text", {"granularity": "global", "extractor": "initial"}, {}),
        ("per_question_cache", {"extractor": "initial"}, {}),
        ("periodic_refresh", {"refresh_interval": 5}, {}),
        ("global_current", {"granularity": "global"}, {}),
        ("global_periodic_refresh", {"granularity": "global", "refresh_interval": 5}, {}),
        ("wrong_question", {"extractor": "initial", "vector_assignment": "other_question"}, {}),
        ("negative", {"direction_control": "negative"}, {}),
        ("random_vector", {"direction_control": "random"}, {}),
        ("random_gate", {"gate": "random", "random_probability": calibration["overlap"]["observed_rate"]}, {}),
        ("matched_random", {"gate": "matched_random", "random_probability": calibration["overlap"]["observed_rate"]}, {}),
        ("matched_position_random", {"gate": "matched_position_random", "random_probability": calibration["overlap"]["observed_rate"]}, {}),
        ("late_positions", {"gate": "late", "start_token": calibration["late_start_token"]}, {}),
        ("last_10_percent_off", {}, {"assistance_until_step": 45}),
        ("phase_both", {}, {"loss_objective": "candidate_rkl"}),
        ("phase_rollout_only", {}, {"steering_phase": "rollout_only", "loss_objective": "candidate_rkl"}),
        # No probe on a plain rollout: this contrast uses all learning positions.
        ("phase_learning_only_all", {"gate": "all"}, {"steering_phase": "learning_only", "loss_objective": "candidate_rkl"}),
        ("phase_both_all", {"gate": "all"}, {"loss_objective": "candidate_rkl"}),
        ("phase_vanilla", {}, {"groups": ["vanilla"], "loss_objective": "candidate_rkl"}),
        ("compass_candidate", {}, {"groups": ["vanilla"], "loss_objective": "compass_candidate"}),
        ("compass_sampled", {}, {"groups": ["vanilla"], "loss_objective": "compass_sampled"}),
        ("eopd_top16_surrogate", {}, {"groups": ["vanilla"], "loss_objective": "eopd_top16_surrogate"}),
        ("sampled_rkl", {}, {"groups": ["vanilla"], "loss_objective": "sampled_rkl"}),
        ("eopd_sampled", {}, {"groups": ["vanilla"], "loss_objective": "eopd_sampled"}),
        ("training_compute_matched", {}, {"groups": ["vanilla"], "steps": 200, "eval_steps": [],
            "dev_eval_steps": [], "match_training_gpu_seconds_to": f"{root}/main_222/seed42", "minimum_steps": 50}),
        ("training_compute_matched_vllm", {"gate": "all"}, {"groups": ["vanilla"], "generation_backend": "vllm",
            "steps": 200, "eval_steps": [], "dev_eval_steps": [],
            "match_training_gpu_seconds_to": f"{root}/main_222/seed42", "minimum_steps": 50}),
        ("total_compute_matched_vllm", {"gate": "all"}, {"groups": ["vanilla"], "generation_backend": "vllm",
            "steps": 400, "eval_steps": [], "dev_eval_steps": [],
            "match_training_gpu_seconds_to": f"{root}/main_222/seed42", "minimum_steps": 50,
            "match_extra_cost_file": f"{root}/preparation_costs.json"}),
    ]
    for direction in ("T-S", "T-R", "R-S"):
        if direction != "T-R":  # global_same_text already IS this experiment.
            definitions.append((f"common_global_{direction}", {"direction": direction, "granularity": "global", "extractor": "initial"}, {}))
            definitions.append((f"common_cached_{direction}", {"direction": direction, "extractor": "initial"}, {}))
        definitions.append((f"both_correct_global_{direction}", {"direction": direction, "granularity": "global", "extractor": "initial"},
            {"reference_pool_dir": f"{root}/references/both_correct",
             "direction_questions_file": f"{root}/references/both_correct/questions.json"}))
    definitions.append(("template_length_control", {"granularity": "global", "extractor": "initial"},
        {"reference_pool_dir": f"{root}/references/template_length",
         "direction_questions_file": f"{root}/references/template_length/questions.json"}))
    for value in calibration.get("overlap_candidates", []):
        if value["target_rate"] != .2:
            definitions.append((f"overlap_rate_{int(100*value['target_rate'])}", {"overlap_threshold": value["threshold"]}, {}))
    definitions.append(("independent_global", {"granularity": "global", "extractor": "initial"},
        {"reference_pool_dir": f"{root}/references/independent_common",
         "direction_questions_file": f"{root}/references/independent_common/questions.json"}))
    for direction, condition in (per_direction_selection or {}).items():
        if direction != "T-R" and (condition["layers"] != selection["layers"] or condition["rho"] != selection["rho"]):
            definitions.append((f"direction_{direction.replace('-', '')}_tuned", {"direction": direction,
                "layers": condition["layers"], "rho": condition["rho"]}, {}))
    results = []
    for name, changes, options in definitions:
        seeds = (42, 43, 44) if name in ("vanilla", "main_222", "random_vector") else (42,)
        for seed in seeds:
            cfg = copy.deepcopy(base)
            cfg.update(id=f"full_{name}_seed{seed}", authorized=True, execution_hold=None,
                run_dir=f"{root}/{name}/seed{seed}", data_dir=f"data/full_scheme_20261112/{name}/seed{seed}",
                groups=["steer"], seed=seed, reference="assets/models/Qwen3-0.6B", generation_backend="hf_batched",
                steering_phase="both", loss_objective="reference_surrogate", steps=50,
                eval_steps=[50], dev_eval_steps=[10, 25, 50],
                shared_reference_dir=f"{root}/references/raw", reference_pool_dir=f"{root}/references/common",
                common_training_ids_file=f"{root}/references/common/manifest.json",
                fixed_activation_scales=f"{root}/precheck/directions/fixed_scales.pt",
                cache_token_activations=False, minimum_direction_pairs=8,
                minimum_free_gib=40,
                retain_model_steps=[10, 25, 50] if name in ("vanilla", "main_222") else [50],
                shared_initial_pool=f"{root}/references/initial_pools",
                evaluation_cache_source="runs/figure8_shared_top16_v2_20261112",
                additional_evaluation_sets=["math500"],
                injection="recorded_causal_gate_mask_rollout_and_replay",
                first_experiment_anchor="T-S/all/S0-global precheck; main panel is separately named")
            cfg.pop("steering_method", None)
            cfg.update(options)
            cfg["steering_method"] = {**method, **changes, "seed": seed}
            cfg.update(direction=cfg["steering_method"]["direction"], layer=cfg["steering_method"]["layers"][0],
                rho=cfg["steering_method"]["rho"],
                implementation="single_gpu_exact_causal_batched_hf_with_vllm_reference_generation_and_eval",
                deviations=["100 candidate OT3 prompts; all main branches share the exact common-valid T/R/S training subset",
                    "50 updates per standard run; compute-matched vanilla has explicit200/400-update safety caps and measured stop",
                    "Shared Base initialization; same-vocabulary151665-output support; no local SFT",
                    "Primary gate comparisons use the same batched HF backend; named vLLM controls report backend effects separately",
                    "AIME2025 avg@16 and MATH500 avg@1 at final checkpoint; independent dev96 at10/25/50",
                    "Compass/EOPD objectives use the same pilot data, model, optimizer and budget; not full paper reproduction"])
            Method(**{**cfg["steering_method"], "layers": tuple(cfg["steering_method"]["layers"])})
            results.append((name, seed, cfg))
    first = [value for value in results if value[0] in ("vanilla", "main_222")]
    first.sort(key=lambda value: (value[1], value[0] != "vanilla"))
    return first + [value for value in results if value[0] not in ("vanilla", "main_222")]


def gpu_seconds(run, training_only=True, phase_prefixes=None):
    total = 0.
    for path in Path(run).glob("attempts/*/process.json"):
        record = read_json(path)
        phase = path.parent.name
        if phase_prefixes is not None and not any(phase.startswith(prefix) for prefix in phase_prefixes):
            continue
        if training_only and not any(label in phase for label in ("rollout", "_train_", "_teacher_", "direction_", "calibration", "bootstrap", "_auxiliary_")):
            continue
        if record.get("finished_at"):
            seconds = (datetime.fromisoformat(record["finished_at"])-datetime.fromisoformat(record["started_at"])).total_seconds()
            total += max(0., seconds) * len(record["gpus"])
    return total


def paired_accuracy_interval(left, right, seed=42):
    ids = sorted(set(left) & set(right))
    if not ids or set(left) != set(right):
        raise ValueError("Paired accuracy needs identical question sets")
    values = np.array([left[q]-right[q] for q in ids])
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(ids), size=(5000, len(ids)))].mean(-1)
    return {"delta_pp": float(values.mean()*100), "paired_question_95ci_pp": (np.quantile(samples,[.025,.975])*100).tolist(),
            "questions": len(ids)}
