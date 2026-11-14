"""Versioned CPU job assembly for the new-pair, zero-update eight-way screen."""

import random

import torch

from steer_opd.compass_alignment import METRIC
from steer_opd.compass_job import load_codecs, prepare_plans, project_path, model_path, safe_key, native_score_binding
from steer_opd.compass_scoring import AlignedMetricAdapter
from steer_opd.direction_bank import file_sha256
from steer_opd.execution_policy import execution_policy, require_default_scheduler_options
from steer_opd.overlap_tuning import calibrate_threshold
from steer_opd.math_data import prompt_ids
from steer_opd.resumable_core import atomic_json, digest, read_json
from steer_opd.rollout_topk import load_rollout_topk


def frozen_json(path, value):
    if path.exists() and read_json(path) != value:
        raise ValueError(f"Refuse to change frozen artifact: {path}")
    atomic_json(path, value)


def validate_config(cfg):
    require_default_scheduler_options(cfg)
    if cfg.get("seed")!=42: raise ValueError("Current experiment fixes seed=42")
    expected = {"teacher": "assets/models/Qwen3.5-35B-A3B", "reference": "assets/models/Qwen3.5-0.8B", "student": "assets/models/Qwen3-4B"}
    if any(cfg.get(k) != v for k, v in expected.items()):
        raise ValueError("This campaign is the explicitly selected Qwen3.5 T/Ref and Qwen3-4B student pair")
    if cfg.get("optimizer_updates") != 0:
        raise ValueError("The selected screen requires zero updates")
    from steer_opd.operators import validate_operator
    validate_operator('slerp',cfg['alpha'])
    layers=cfg['layers']
    from steer_opd.resumable_core import read_json
    blocks=read_json(model_path(cfg['student'])/'config.json')['num_hidden_layers']
    if len(layers)!=1 or type(layers[0]) is not int or not 0<=layers[0]<blocks:
        raise ValueError(f'Select one student layer in 0..{blocks-1}')
    if cfg.get("metric") != METRIC or cfg.get("screen_backend") not in ("hf_batched_static", "easysteer"):
        raise ValueError("Explicitly select the aligned project metric and an available backend")
    if cfg.get("gate_metric", "overlap") not in ("overlap", "P_overlap") or cfg.get("rank_metric", "overlap") not in ("overlap", "P_overlap"):
        raise ValueError("Select one explicit static gate/ranking metric; both metrics are always recorded")
    if (not 0 < cfg["select_fraction"] < 1 or cfg["minimum_split_questions"] < 1
            or not 1 <= cfg["confirm_top_n"] <= len(scheme_conditions(cfg)) or not 0 < cfg["gate_target_rate"] < 1
            or cfg["teacher_gpus"] not in (1, 2)):
        raise ValueError("Invalid split/calibration/confirmation/GPU budget")


def scheme_conditions(cfg):
    conditions=[]
    for direction in ('T-S','T-R'):
        for gate in ('all','static_overlap'):
            for granularity in ('global','per_question'):
                conditions.append({'id':f'scheme_{len(conditions)+1:02d}','direction':direction,
                    'gate':gate,'granularity':granularity,'bank':f'{direction}_{granularity}',
                    'operator':'slerp','alpha':cfg['alpha'],'layers':cfg['layers']})
    selected=cfg.get('scheme_ids',[c['id'] for c in conditions])
    if (not selected or len(set(selected))!=len(selected)
            or set(selected)-{c['id'] for c in conditions}):
        raise ValueError('Select unique scheme IDs from scheme_01..scheme_08')
    return [c for c in conditions if c['id'] in selected]


def prepare_questions(cfg, codecs):
    original = read_json(project_path(cfg["question_source"]))
    questions, rejected = [], {}
    for row in original:
        prompts = {role: prompt_ids(codec.tokenizer, row["question"], enable_thinking=cfg["enable_thinking"])
                   for role, codec in codecs.items()}
        if any(len(p) > cfg["max_prompt_tokens"] for p in prompts.values()):
            rejected[row["id"]] = "native_prompt_exceeds_frozen_limit"
            continue
        questions.append({"id": row["id"], "question": row["question"], "answer": row["answer"],
                          "student_prompt_ids": prompts["student"]})
    if len(questions) < 2*cfg["minimum_split_questions"]:
        raise ValueError("Too few shared prompt-valid training questions")
    shuffled = [q["id"] for q in questions]
    random.Random(cfg["seed"]).shuffle(shuffled)
    cut = int(len(shuffled)*cfg["select_fraction"])
    if min(cut, len(shuffled)-cut) < cfg["minimum_split_questions"]:
        raise ValueError("Insufficient predeclared select/confirm questions")
    data = project_path(cfg["data_dir"])
    frozen_json(data / "questions.json", questions)
    frozen_json(data / "split.json", {"select_ids": shuffled[:cut], "confirm_ids": shuffled[cut:],
        "seed": cfg["seed"], "source_sha256": file_sha256(project_path(cfg["question_source"])),
        "rejected_prompts": rejected, "uses_test": False, "scope": "training_questions_with_privileged_reference_vectors_not_independent_dev"})
    return questions


def generation_job(cfg, codec, role, questions, *, source=False):
    output = f"{cfg['run_dir']}/source" if source else f"{cfg['run_dir']}/references/{role}"
    requests = [{"id": q["id"], "key": f"{q['id']}_0" if source else q["id"], "question": q["question"],
                 "answer": q["answer"], "seed": cfg["seed"],
                 "prompt_token_ids": prompt_ids(codec.tokenizer, q["question"], enable_thinking=cfg["enable_thinking"])}
                for q in questions]
    job = {**execution_policy(cfg), "model": cfg[role], "output": output, "requests": requests, "vector": None,
        "max_tokens": cfg["source_max_tokens"] if source else cfg["reference_max_tokens"],
        "temperature": 1. if source else cfg["reference_temperature"], "top_p": 1. if source else cfg["reference_top_p"],
        "seed": cfg["seed"], "support_size": codec.support_size, "gpu_memory_utilization": cfg["gpu_memory_utilization"],
        "logprobs": 16 if source else None, "head_dtype": "float32", "normalization": "native",
        "backend_version": 6, "decode_cuda_graph": cfg.get("decode_cuda_graph", True),
        "tensor_parallel_size": cfg["teacher_gpus"] if role == "teacher" else 1}
    job.update({k: cfg[k] for k in ("student", "teacher", "reference", "tokenizer_paths")})
    job.update(role=role, enable_prefix_caching=True)
    job["reproducibility_protocol"] = 1
    job["resume_granularity"] = "engine_call"
    job["bind_checkpoint_content"] = cfg.get("bind_checkpoint_content", False)
    if codec.valid_ids != list(range(codec.support_size)):
        if source:
            raise ValueError("Source native raw-Top16 profile requires contiguous student IDs")
        job["allowed_token_ids"] = codec.valid_ids
    if source:
        job.update(save_rollout_top_k=16, logprobs_mode="raw_logprobs", record_token_entropy=True)
    return job


def make_base_job(cfg, codecs, rows):
    run = cfg["run_dir"]
    banks = {f"{d}_{g}": f"{run}/banks/{d}_{g}.pt" for d in ("T-S", "T-R") for g in ("global", "per_question")}
    conditions = scheme_conditions(cfg)
    job = {k: cfg[k] for k in ("student", "teacher", "reference", "enable_thinking", "tokenizer_paths")}
    job.update(**execution_policy(cfg), stage="static_overlap_screen", optimizer_updates=0, rows=rows,
        questions=f"{cfg['data_dir']}/questions.json", source_dir=f"{run}/source", support_size=codecs["student"].support_size,
        teacher_scores=f"{run}/native_scores/teacher", reference_scores=f"{run}/native_scores/reference",
        alignment_dir=f"{run}/alignment", banks=banks, conditions=conditions,
        identity={"metric": METRIC}, backend=cfg["screen_backend"], batch_size=cfg["student_score_batch_size"],
        teacher_batch_size=cfg["teacher_score_batch_size"], reference_batch_size=cfg["reference_score_batch_size"],
        gpu_memory_utilization=cfg["gpu_memory_utilization"], chunk=cfg["logit_chunk_size"])
    job.update(require_dual_metrics=cfg.get("require_dual_metrics", True),
        seed=cfg["seed"], reproducibility_protocol=1, resume_granularity="engine_call",
        bind_checkpoint_content=cfg.get("bind_checkpoint_content", False),
        gate_metric=cfg.get("gate_metric", "overlap"), rank_metric=cfg.get("rank_metric", "overlap"),
        probability_backend=cfg.get("probability_backend", "easysteer"), teacher_gpus=cfg["teacher_gpus"])
    return job


def prepare_initial_jobs(cfg, codecs):
    questions = prepare_questions(cfg, codecs)
    run = project_path(cfg["run_dir"])
    for role in ("teacher", "reference", "student"):
        frozen_json(run / "jobs" / f"reference_{role}.json", generation_job(cfg, codecs[role], role, questions))
    bank_cfg = {**cfg, "questions": f"{cfg['data_dir']}/questions.json", "bank_dir": f"{cfg['run_dir']}/banks",
        "reference_dirs": {r: f"{cfg['run_dir']}/references/{r}" for r in ("teacher", "reference", "student")},
        "minimum_pairs": 2*cfg["minimum_split_questions"]}
    frozen_json(run / "jobs/banks.json", bank_cfg)
    frozen_json(run / "jobs/alignment_and_scoring.json", make_base_job(cfg, codecs, f"{cfg['run_dir']}/source_rows.json"))
    return questions


def prepare_screen_jobs(cfg, base_job, codecs):
    """Calibrate only on frozen select questions, then freeze 8-way jobs."""
    rows, plans, role_rows, _ = prepare_plans(base_job, codecs)
    split = read_json(project_path(cfg["data_dir"]) / "split.json")
    select_ids, confirm_ids = set(split["select_ids"]), set(split["confirm_ids"])
    selected = [r for r in rows if r["id"] in select_ids]
    confirmed = [r for r in rows if r["id"] in confirm_ids]
    if min(len({r["id"] for r in selected}), len({r["id"] for r in confirmed})) < cfg["minimum_split_questions"]:
        raise ValueError("Too few common usable questions in select/confirm; split is not silently changed")
    teachers, refs = {}, {}
    native_rows = {role: {r["key"]: r for r in values} for role, values in role_rows.items()}
    for row in selected:
        key = safe_key(row)
        teachers[key] = torch.load(project_path(base_job["teacher_scores"]) / f"{key}.pt", weights_only=True, map_location="cpu")
        refs[key] = torch.load(project_path(base_job["reference_scores"]) / f"{key}.pt", weights_only=True, map_location="cpu")
        for role, record in [("teacher", teachers[key]), ("reference", refs[key])]:
            expected = native_score_binding(base_job, role, native_rows[role][key], codecs[role])
            if record.get("binding") != expected:
                raise ValueError("Native source scoring identity changed before threshold calibration")
            if base_job.get("require_dual_metrics"):
                from steer_opd.byte_probabilities import projection_spec, validate_projection
                validate_projection(record, projection_spec(codecs[role], codecs["student"]),
                                    len(native_rows[role][key]["token_ids"]))
    adapter = AlignedMetricAdapter(plans, codecs["student"], codecs["teacher"], refs,
                                   require_dual=base_job.get("require_dual_metrics", False))
    names = ("overlap", "P_overlap") if base_job.get("require_dual_metrics") else ("overlap",)
    values = {name: [] for name in names}
    for row in selected:
        if row.get("model") != base_job["student"]:
            raise ValueError("Threshold source came from a different student")
        score = load_rollout_topk(project_path(base_job["source_dir"]), row, 16, base_job["support_size"])
        metric = adapter(safe_key(row), score, teachers[safe_key(row)])
        for name in names:
            values[name].extend(metric[name][metric["alignment_mask"]].double().tolist())
    calibrations = {name: calibrate_threshold(v, cfg["gate_target_rate"], "lt", lower_bound=0.) for name, v in values.items()}
    calibration = calibrations[cfg.get("gate_metric", "overlap")]
    if not 0 <= calibration["threshold"] <= 1:
        raise ValueError("Calibrated overlap threshold outside the declared [0,1] domain")
    run = project_path(cfg["run_dir"])
    frozen_json(run / "thresholds.json", {"metrics": calibrations, "gate_metric": cfg.get("gate_metric", "overlap"),
                                        "uses_confirm": False, "uses_test": False})
    frozen_json(run / "threshold.json", {**calibration, "metric": cfg.get("gate_metric", "overlap"), "select_question_ids": sorted({r['id'] for r in selected}),
                                         "uses_confirm": False, "uses_test": False})
    frozen_json(run / "select_rows.json", selected)
    frozen_json(run / "confirm_rows.json", confirmed)
    select_job = {**base_job, "rows": f"{cfg['run_dir']}/select_rows.json", "alignment_dir": f"{cfg['run_dir']}/select_alignment",
                  "threshold": calibration["threshold"], "output": f"{cfg['run_dir']}/select"}
    # Native scoring rows depend on question/native tokens, not which split
    # later consumes them; alignment IDs are preserved between these plans.
    confirm_job = {**base_job, "rows": f"{cfg['run_dir']}/confirm_rows.json", "alignment_dir": f"{cfg['run_dir']}/confirm_alignment",
                   "threshold": calibration["threshold"], "output": f"{cfg['run_dir']}/confirm"}
    frozen_json(run / "jobs/select.json", select_job)
    return select_job, confirm_job
