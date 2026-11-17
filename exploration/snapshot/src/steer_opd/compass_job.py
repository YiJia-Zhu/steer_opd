"""Prepare and validate the cross-tokenizer data contract for the T-S screen."""

from pathlib import Path

import torch

from steer_opd.compass_alignment import ByteLevelCodec, METRIC, align_response
from steer_opd.compass_scoring import AlignedMetricAdapter, AlignedTeacherRecords
from steer_opd.direction_bank import file_sha256
from steer_opd.resumable_core import ROOT, atomic_json, atomic_torch, digest, read_json
from steer_opd.rollout_topk import load_rollout_topk
from steer_opd.static_masks import freeze_static_mask, validate_native_score

MODEL_ROLES = ('student','teacher')
SCORING_ROLES = ('teacher',)


def project_path(value):
    from steer_opd.artifact_paths import relocate_path
    p = Path(value)
    p = relocate_path(p if p.is_absolute() else ROOT/p)
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
    from steer_opd.prefix_matching import METRIC_VERSION, RULE_VERSION, codec_identity
    identity = {"row": row, "checkpoint": job[role], "tokenizer": codec.mapping_sha256,
                "support_id": codec.support_id, "metric_version": METRIC_VERSION,
                "matching_rule_version": RULE_VERSION, "token_event_identity": codec_identity(codec), "returned_top_k": 16}
    if job.get("require_dual_metrics"):
        identity.update(protocol="top16_raw_dual_overlap_v4", backend=job.get("probability_backend"))
    if job.get("bind_checkpoint_content"):
        from steer_opd.checkpoint_identity import checkpoint_content_id
        identity["checkpoint_content_id"] = checkpoint_content_id(model_path(job[role]),
            receipt_dir=project_path(job[f"{role}_scores"])/"provenance/checkpoints")
    return digest(identity)


def load_codecs(job):
    from transformers import AutoTokenizer
    return {role: ByteLevelCodec(AutoTokenizer.from_pretrained(
            model_path(job.get("tokenizer_paths", {}).get(role, job[role])), local_files_only=True))
            for role in MODEL_ROLES}


def cached_source_metrics(job, row, student, teacher, adapter, student_sha256, teacher_sha256):
    """Share source metrics between calibration and select/confirm mask building.

    The cache belongs to the original unsteered source, independent of split,
    threshold and candidate. A checksum marker commits each CPU-derived record;
    probability/checkpoint validation remains the caller's responsibility.
    """
    from steer_opd.prefix_matching import METRIC_VERSION, RULE_VERSION, P_DEFINITION, TOPK_TIE_POLICY
    key = safe_key(row)
    plan = adapter.plans[key]
    identity = {'schema':'source_dual_metrics_v1', 'row_sha256':digest(row),
        'student_score_sha256':student_sha256, 'teacher_score_sha256':teacher_sha256,
        'alignment_id':plan['alignment_id'], 'metric_version':METRIC_VERSION,
        'matching_rule_version':RULE_VERSION, 'P_overlap_definition':P_DEFINITION,
        'topk_tie_policy':TOPK_TIE_POLICY,
        'metric_implementation':{name:file_sha256(Path(__file__).with_name(name))
            for name in ('compass_alignment.py','prefix_matching.py')}}
    binding = digest(identity)
    folder = project_path(job['source_metrics_dir']) if job.get('source_metrics_dir') else project_path(job['source_dir']).parent/'alignment'
    path = folder/f'{key}.metrics.pt'
    marker = folder/f'{key}.metrics.json'
    if marker.is_file():
        committed = read_json(marker)
        if committed.get('binding') != binding:
            raise ValueError('Source metric cache inputs or definition changed')
        if not path.is_file() or file_sha256(path) != committed['sha256']:
            raise ValueError('Source metric cache checksum changed')
        packet = torch.load(path, map_location='cpu', weights_only=True)
        if packet.get('binding') != binding:
            raise ValueError('Source metric cache binding changed')
        return packet['metrics']
    metrics = adapter(key, student, teacher)
    # Source gates need two metric arrays and their identities, not the larger
    # per-span matching/likelihood diagnostics returned by candidate scoring.
    fields = ('overlap','P_overlap','alignment_mask','alignment_id','metric_version',
              'matching_rule_version','P_overlap_definition','topk_tie_policy')
    saved = {name:metrics[name] for name in fields}
    atomic_torch(path, {'binding':binding, 'metrics':saved})
    atomic_json(marker, {'binding':binding, 'identity':identity, 'sha256':file_sha256(path)})
    return saved


def prepare_plans(job, codecs=None):
    """CPU-only native retokenization, prompts and immutable common boundaries."""
    from steer_opd.math_data import prompt_ids
    codecs = load_codecs(job) if codecs is None else codecs
    rows = read_json(project_path(job["rows"]))
    questions = {q["id"]: q for q in read_json(project_path(job["questions"]))}
    if not rows or len({safe_key(r) for r in rows}) != len(rows):
        raise ValueError("Need nonempty source rows with unique response keys")
    plans, role_rows, rejected = {}, {role: [] for role in SCORING_ROLES}, {}
    output = project_path(job["alignment_dir"])
    cache=job.get('alignment_source_dir')
    if cache:
        folder=project_path(cache);manifest=read_json(folder/'manifest.json')
        expected={r:codecs[r].mapping_sha256 for r in MODEL_ROLES}
        if (manifest['tokenizer_mapping_sha256']!=expected
                or manifest['question_sha256']!=file_sha256(project_path(job['questions']))
                or manifest['enable_thinking']!=job.get('enable_thinking',False)):
            raise ValueError('Shared alignment question/tokenizer/template identity changed')
        for row in rows:
            key=safe_key(row)
            if row.get('steering') or any(row.get('steering_mask',[])):
                raise ValueError('Alignment cache requires an unsteered source')
            if row['prompt_token_ids']!=prompt_ids(codecs['student'].tokenizer,questions[row['id']]['question'],
                                                  enable_thinking=job.get('enable_thinking',False)):
                raise ValueError('Shared alignment source prompt changed')
            if key not in manifest['valid_source_keys']:continue
            plan=read_json(folder/f'{key}.json')
            if (plan['source_token_ids']!=row['token_ids'] or plan['tokenizer_mapping_sha256']!=expected
                    or plan['alignment_id']!=digest({k:v for k,v in plan.items() if k!='alignment_id'})):
                raise ValueError('Shared alignment source tokens or plan changed')
            plans[key]=plan
        for role in role_rows:
            saved={r['key']:r for r in read_json(folder/f'{role}_rows.json')}
            role_rows[role]=[saved[safe_key(r)] for r in rows if safe_key(r) in plans]
        return [r for r in rows if safe_key(r) in plans],plans,role_rows,codecs
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
                                  {r: codecs[r] for r in SCORING_ROLES})
        except (UnicodeError, ValueError) as error:
            rejected[key] = str(error)
            continue
        if not any(plan["alignment_mask"]):
            rejected[key] = "no_consistent_noncontrol_text_span"
            continue
        plan_path = output / f"{key}.json"
        if plan_path.exists() and read_json(plan_path) != plan:
            raise ValueError("Frozen alignment tokens/tokenizer/bytes changed")
        if not plan_path.exists():atomic_json(plan_path, plan)
        plans[key] = plan
        for role in role_rows:
            prompt = prompt_ids(codecs[role].tokenizer, question["question"],
                                enable_thinking=job.get("enable_thinking", False))
            role_rows[role].append({"id": row["id"], "key": key, "prompt_token_ids": prompt,
                                   "token_ids": plan["native_response_ids"][role],
                                   "alignment_id": plan["alignment_id"]})
    manifest = {"source_rows_sha256": file_sha256(project_path(job["rows"])),
                "question_sha256": file_sha256(project_path(job["questions"])),
                "tokenizer_mapping_sha256": {r: codecs[r].mapping_sha256 for r in MODEL_ROLES},
                "valid_source_keys": list(plans), "rejected": rejected,
                "role_rows_hash": digest(role_rows), "enable_thinking": job.get("enable_thinking", False)}
    if (output / "manifest.json").exists() and read_json(output / "manifest.json") != manifest:
        raise ValueError("Cross-tokenizer source alignment manifest changed")
    if not (output/"manifest.json").exists():atomic_json(output/"manifest.json",manifest)
    for role,values in role_rows.items():
        path=output/f"{role}_rows.json"
        if path.exists():
            if read_json(path)!=values:raise ValueError("Native alignment rows changed")
        else:atomic_json(path,values)
    return [r for r in rows if safe_key(r) in plans], plans, role_rows, codecs


def prepare_screen(job, codecs=None):
    """Reuse source Top16 and teacher likelihood caches without model calls."""
    from steer_opd.method_config import Method
    from steer_opd.static_screen import candidate_vectors
    from steer_opd.strength_schedule import row_strength,validate_schedule
    if job.get("stage") != "static_overlap_screen" or job.get("optimizer_updates") != 0 or job["identity"]["metric"] != METRIC:
        raise ValueError("Expected an explicit zero-update aligned-byte metric screen")
    rows, plans, role_rows, codecs = prepare_plans(job, codecs)
    if not rows:
        raise ValueError("No jointly alignable sources")
    banks = {k: torch.load(project_path(p), weights_only=True, map_location="cpu") for k, p in job["banks"].items()}
    from steer_opd.method_config import DIRECTION_VERSION, DIRECTION_VERSIONS, reader_direction
    steer_role=job.get('steer_role','student');direction=reader_direction(steer_role)
    version=DIRECTION_VERSIONS[direction]
    if (job.get('direction_version',DIRECTION_VERSION)!=version
            or any(b['method']['direction']!=direction or b['identity'].get('direction_version')!=version
                   for b in banks.values())):
        raise ValueError('Direction bank differs from the intervention reader/space protocol')
    conditions = job["conditions"]
    schedule=job.get('strength_schedule')
    if schedule is not None:validate_schedule(schedule)
    if len({c["id"] for c in conditions}) != len(conditions):
        raise ValueError("Duplicate candidate IDs")
    for c in conditions:
        if c.get('steer_role','student')!=steer_role or c['direction']!=direction:
            raise ValueError('A GPU screen must contain a single intervention role')
        if c["id"] in (".", "..", "baseline", "masks") or Path(c["id"]).name != c["id"]:
            raise ValueError("Unsafe candidate ID")
        if c["gate"] not in ("all", "static_overlap"):
            raise ValueError("Expected all or frozen static_overlap")
        Method(direction=c["direction"], granularity=c["granularity"], layers=tuple(c["layers"]),
               gate=c["gate"], overlap_threshold=job["threshold"], operator="slerp", alpha=c["alpha"])
    native = {role: {} for role in SCORING_ROLES}
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
            native[role][key] = value
            hashes[role][key] = file_sha256(p)
    adapter = AlignedMetricAdapter(plans, codecs["student"], codecs["teacher"],
                                   require_dual=job.get("require_dual_metrics", False))
    masks = {}
    source_hashes = {};source_scores={}
    for row in rows:
        key = safe_key(row)
        source_root = project_path(job["source_dir"])
        score = load_rollout_topk(source_root, row, 16, job["support_size"])
        if row.get("model") != job["student"] or job["support_size"] != codecs["student"].support_size:
            raise ValueError("Source rollout model/native output support mismatch")
        if job.get("bind_checkpoint_content"):
            from steer_opd.checkpoint_identity import checkpoint_content_id
            content_id = checkpoint_content_id(model_path(job["student"]))
            reader_content=checkpoint_content_id(model_path(job[steer_role]))
            if row.get("model_content_id") != content_id or any(
                    bank["identity"].get("checkpoint_content_id") != reader_content for bank in banks.values()):
                raise ValueError("Source or intervention-reader checkpoint contents changed")
        source_scores[key]=score
        source_hashes[key] = file_sha256(source_root / row["rollout_topk"])
        metrics = cached_source_metrics(job, row, score, native['teacher'][key], adapter,
                                        source_hashes[key], hashes['teacher'][key])
        identity = {**{k:v for k,v in job["identity"].items() if k not in ("direction_version","direction_versions")}, "student_checkpoint": job["student"], "teacher_checkpoint": job["teacher"],
            "student_tokenizer_sha256": codecs["student"].mapping_sha256,
            "teacher_tokenizer_sha256": codecs["teacher"].mapping_sha256,
            "support_id": digest({r: codec.support_id for r, codec in codecs.items()}),
            "student_score_sha256": source_hashes[key], "teacher_score_sha256": hashes["teacher"][key],
            "alignment_id": plans[key]["alignment_id"]}
        identity.update(metric_version=metrics["metric_version"], matching_rule_version=metrics["matching_rule_version"],
                        P_overlap_definition=metrics["P_overlap_definition"], topk_tie_policy=metrics["topk_tie_policy"])
        gate_metric = job.get("gate_metric", "overlap")
        identity["gate_metric"] = gate_metric
        masks[key] = freeze_static_mask(row, metrics[gate_metric], job["threshold"], identity,
                                       alignment_mask=plans[key]["alignment_mask"],
                                       source_metrics={name: metrics[name] for name in ("overlap", "P_overlap")}
                                       if job.get("require_dual_metrics") else None)
        for c in conditions:
            bank = banks[c["bank"]]
            if row["id"] not in bank["valid_ids"] or bank["identity"]["checkpoint"] != job[steer_role]:
                raise ValueError("All schemes must share source questions and the same extractor")
            strength=row_strength(c,row,schedule)
            if strength['effective_alpha']!=0:candidate_vectors(bank,row,{**c,'alpha':strength['effective_alpha']})
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
    records=AlignedTeacherRecords(native['teacher'],adapter)
    records.native_rows={r['key']:r for r in role_rows['teacher']}
    records.source_scores=source_scores
    return rows, masks, records, banks
