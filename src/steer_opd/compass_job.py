"""Prepare and validate the cross-tokenizer data contract for the 8-way screen."""

from pathlib import Path

import torch

from steer_opd.compass_alignment import ByteLevelCodec, METRIC, align_response
from steer_opd.compass_scoring import AlignedMetricAdapter, AlignedTeacherRecords
from steer_opd.direction_bank import file_sha256
from steer_opd.resumable_core import ROOT, atomic_json, digest, read_json
from steer_opd.rollout_topk import load_rollout_topk
from steer_opd.static_masks import freeze_static_mask, validate_native_score


def project_path(value):
    p = Path(value)
    p = p if p.is_absolute() else ROOT / p
    if not p.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("Cross-tokenizer artifacts must stay in the project")
    return p


def model_path(value):
    """Read-only assets may be project links to the shared weight directory."""
    p = Path(value)
    p = p if p.is_absolute() else ROOT / p
    if p.is_relative_to(ROOT / "assets/models"):
        return p
    return project_path(value)


def safe_key(row):
    key = row.get("key", row["id"])
    if not isinstance(key, str) or Path(key).name != key or key in (".", ".."):
        raise ValueError("Unsafe response key")
    return key


def native_score_binding(job, role, row, codec):
    identity = {"row": row, "checkpoint": job[role], "tokenizer": codec.mapping_sha256,
                "support_id": codec.support_id}
    if job.get("require_dual_metrics"):
        identity.update(protocol="dual_overlap_v2", backend=job.get("probability_backend"))
    if job.get("bind_checkpoint_content"):
        from steer_opd.checkpoint_identity import checkpoint_content_id
        identity["checkpoint_content_id"] = checkpoint_content_id(model_path(job[role]),
            receipt_dir=project_path(job[f"{role}_scores"])/"provenance/checkpoints")
    return digest(identity)


def load_codecs(job):
    from transformers import AutoTokenizer
    return {role: ByteLevelCodec(AutoTokenizer.from_pretrained(
            model_path(job.get("tokenizer_paths", {}).get(role, job[role])), local_files_only=True))
            for role in ("student", "teacher", "reference")}


def prepare_plans(job, codecs=None):
    """CPU-only native retokenization, prompts and immutable common boundaries."""
    from steer_opd.math_data import prompt_ids
    codecs = load_codecs(job) if codecs is None else codecs
    rows = read_json(project_path(job["rows"]))
    questions = {q["id"]: q for q in read_json(project_path(job["questions"]))}
    if not rows or len({safe_key(r) for r in rows}) != len(rows):
        raise ValueError("Need nonempty source rows with unique response keys")
    plans, role_rows, rejected = {}, {role: [] for role in ("teacher", "reference")}, {}
    output = project_path(job["alignment_dir"])
    for row in rows:
        key = safe_key(row)
        if row.get("steering") or any(row.get("steering_mask", [])):
            raise ValueError("Aligned source must be a whole unsteered trajectory")
        question = questions[row["id"]]
        expected_prompt = prompt_ids(codecs["student"].tokenizer, question["question"],
                                     enable_thinking=job.get("enable_thinking", False))
        if row["prompt_token_ids"] != expected_prompt:
            raise ValueError("Student source prompt differs from the frozen question/template")
        try:
            plan = align_response(row["token_ids"], codecs["student"],
                                  {r: codecs[r] for r in ("teacher", "reference")})
        except (UnicodeError, ValueError) as error:
            rejected[key] = str(error)
            continue
        if not any(plan["alignment_mask"]):
            rejected[key] = "no_consistent_noncontrol_text_span"
            continue
        plan_path = output / f"{key}.json"
        if plan_path.exists() and read_json(plan_path) != plan:
            raise ValueError("Frozen alignment tokens/tokenizer/bytes changed")
        atomic_json(plan_path, plan)
        plans[key] = plan
        for role in role_rows:
            prompt = prompt_ids(codecs[role].tokenizer, question["question"],
                                enable_thinking=job.get("enable_thinking", False))
            role_rows[role].append({"id": row["id"], "key": key, "prompt_token_ids": prompt,
                                   "token_ids": plan["native_response_ids"][role],
                                   "alignment_id": plan["alignment_id"]})
    manifest = {"source_rows_sha256": file_sha256(project_path(job["rows"])),
                "question_sha256": file_sha256(project_path(job["questions"])),
                "tokenizer_mapping_sha256": {r: c.mapping_sha256 for r, c in codecs.items()},
                "valid_source_keys": list(plans), "rejected": rejected,
                "role_rows_hash": digest(role_rows), "enable_thinking": job.get("enable_thinking", False)}
    if (output / "manifest.json").exists() and read_json(output / "manifest.json") != manifest:
        raise ValueError("Cross-tokenizer source alignment manifest changed")
    atomic_json(output / "manifest.json", manifest)
    for role, values in role_rows.items():
        atomic_json(output / f"{role}_rows.json", values)
    return [r for r in rows if safe_key(r) in plans], plans, role_rows, codecs


def prepare_screen(job, codecs=None):
    """Reuse source Top16 and T/R native likelihood caches without model calls."""
    from steer_opd.method_config import Method
    from steer_opd.static_screen import candidate_vectors
    if job.get("stage") != "static_overlap_screen" or job.get("optimizer_updates") != 0 or job["identity"]["metric"] != METRIC:
        raise ValueError("Expected an explicit zero-update aligned-byte metric screen")
    rows, plans, role_rows, codecs = prepare_plans(job, codecs)
    if not rows:
        raise ValueError("No jointly alignable sources")
    banks = {k: torch.load(project_path(p), weights_only=True, map_location="cpu") for k, p in job["banks"].items()}
    conditions = job["conditions"]
    if len({c["id"] for c in conditions}) != len(conditions):
        raise ValueError("Duplicate candidate IDs")
    for c in conditions:
        if c["id"] in (".", "..", "baseline", "masks") or Path(c["id"]).name != c["id"]:
            raise ValueError("Unsafe candidate ID")
        if c["gate"] not in ("all", "static_overlap"):
            raise ValueError("Expected all or frozen static_overlap")
        Method(direction=c["direction"], granularity=c["granularity"], layers=tuple(c["layers"]),
               gate=c["gate"], overlap_threshold=job["threshold"], operator="slerp", alpha=c["alpha"])
    native = {role: {} for role in ("teacher", "reference")}
    hashes = {role: {} for role in native}
    for role in native:
        for native_row in role_rows[role]:
            key = native_row["key"]
            p = project_path(job[f"{role}_scores"]) / f"{key}.pt"
            value = torch.load(p, weights_only=True, map_location="cpu")
            expected = native_score_binding(job, role, native_row, codecs[role])
            if value.get("binding") != expected:
                raise ValueError(f"{role} cache has changed text/checkpoint/tokenizer/support")
            validate_native_score(value, len(native_row["token_ids"]), codecs[role].support_size)
            if job.get("require_dual_metrics"):
                from steer_opd.byte_probabilities import projection_spec, validate_projection
                validate_projection(value, projection_spec(codecs[role], codecs["student"]), len(native_row["token_ids"]))
            native[role][key] = value
            hashes[role][key] = file_sha256(p)
    adapter = AlignedMetricAdapter(plans, codecs["student"], codecs["teacher"], native["reference"],
                                   require_dual=job.get("require_dual_metrics", False))
    masks = {}
    source_hashes = {}
    for row in rows:
        key = safe_key(row)
        source_root = project_path(job["source_dir"])
        score = load_rollout_topk(source_root, row, 16, job["support_size"])
        if row.get("model") != job["student"] or job["support_size"] != codecs["student"].support_size:
            raise ValueError("Source rollout model/native output support mismatch")
        if job.get("bind_checkpoint_content"):
            from steer_opd.checkpoint_identity import checkpoint_content_id
            content_id = checkpoint_content_id(model_path(job["student"]))
            if row.get("model_content_id") != content_id or any(
                    bank["identity"].get("checkpoint_content_id") != content_id for bank in banks.values()):
                raise ValueError("Source/bank checkpoint contents differ from the frozen student")
        source_hashes[key] = file_sha256(source_root / row["rollout_topk"])
        metrics = adapter(key, score, native["teacher"][key])
        identity = {**job["identity"], "student_checkpoint": job["student"], "teacher_checkpoint": job["teacher"],
            "student_tokenizer_sha256": codecs["student"].mapping_sha256,
            "teacher_tokenizer_sha256": codecs["teacher"].mapping_sha256,
            "support_id": digest({r: codec.support_id for r, codec in codecs.items()}),
            "student_score_sha256": source_hashes[key], "teacher_score_sha256": hashes["teacher"][key],
            "reference_score_sha256": hashes["reference"][key], "alignment_id": plans[key]["alignment_id"]}
        gate_metric = job.get("gate_metric", "overlap")
        identity["gate_metric"] = gate_metric
        masks[key] = freeze_static_mask(row, metrics[gate_metric], job["threshold"], identity,
                                       alignment_mask=plans[key]["alignment_mask"],
                                       source_metrics={name: metrics[name] for name in ("overlap", "P_overlap")}
                                       if job.get("require_dual_metrics") else None)
        for c in conditions:
            bank = banks[c["bank"]]
            if row["id"] not in bank["valid_ids"] or bank["identity"]["checkpoint"] != job["student"]:
                raise ValueError("All eight schemes must share source questions and the same extractor")
            candidate_vectors(bank, row, c)
    output = project_path(job["output"])
    frozen = {"job_hash": digest(job), "source_rows_hash": digest(rows),
              "masks": {k: v["mask_id"] for k, v in masks.items()},
              "bank_hashes": {k: file_sha256(project_path(p)) for k, p in job["banks"].items()},
              "aligned_source_keys": list(plans), "score_hashes": hashes, "parameter_updates": 0}
    if (output / "manifest.json").exists() and read_json(output / "manifest.json") != frozen:
        raise ValueError("Cannot resume changed aligned screen inputs")
    for key, mask in masks.items():
        atomic_json(output / "masks" / f"{key}.json", mask)
    atomic_json(output / "manifest.json", frozen)
    return rows, masks, AlignedTeacherRecords(native["teacher"], adapter), banks
