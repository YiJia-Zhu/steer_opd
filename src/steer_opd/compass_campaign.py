"""Versioned CPU jobs for the zero-update fixed/annealed T-S screen."""

import random

import torch

from steer_opd.compass_alignment import METRIC
from steer_opd.prefix_matching import METRIC_VERSION, RULE_VERSION, P_DEFINITION, TOPK_TIE_POLICY
from steer_opd.compass_job import load_codecs, prepare_plans, project_path, model_path, safe_key, native_score_binding, cached_source_metrics, MODEL_ROLES
from steer_opd.compass_scoring import AlignedMetricAdapter
from steer_opd.direction_bank import file_sha256
from steer_opd.execution_policy import execution_policy, require_default_scheduler_options
from steer_opd.overlap_tuning import calibrate_threshold
from steer_opd.math_data import prompt_ids
from steer_opd.resumable_core import ROOT, atomic_json, digest, read_json
from steer_opd.rollout_topk import load_rollout_topk
from steer_opd.method_config import DIRECTION_VERSION, DIRECTION_VERSIONS, TEACHER_DIRECTION_VERSION
from steer_opd.strength_schedule import LINEAR, freeze_schedule, validate_schedule
from steer_opd.stage_io import artifact_dir, metadata_dir


def frozen_json(path, value):
    if path.exists():
        if read_json(path) != value:raise ValueError(f"Refuse to change frozen artifact: {path}")
        return
    atomic_json(path, value)


def validate_config(cfg):
    require_default_scheduler_options(cfg)
    if cfg.get('data_policy','select_confirm') not in ('select_confirm','all_train'):
        raise ValueError('Unknown training question data policy')
    from steer_opd.retention import minimal
    minimal(cfg)
    from steer_opd.generation_policy import presence_penalty
    presence_penalty(cfg)
    from steer_opd.tensor_checkpoint import SUPPORTED_FORMATS
    if 'candidate_tensor_format' in cfg and 'candidate_tensor_compression' in cfg:
        raise ValueError('Choose one candidate tensor format setting')
    storage=cfg.get('candidate_tensor_format',cfg.get('candidate_tensor_compression'))
    if storage is not None and storage not in (*SUPPORTED_FORMATS,'none','gzip'):
        raise ValueError('Unknown candidate tensor format')
    if cfg.get('experiment') not in ('ablation','param','compar') or cfg.get('phase')!='overlap':
        raise ValueError('Only the overlap phase of ablation is implemented; param/baseline remain planned')
    if cfg.get('stop_after','panel') not in ('source_overlap','panel'):
        raise ValueError('Choose source_overlap or panel as the explicit stop point')
    if cfg.get('strength_policies',['fixed'])!=['fixed']:
        raise ValueError('Overlap uses four fixed-strength schemes; annealing is a future training ablation')
    if cfg.get("seed")!=42: raise ValueError("Current experiment fixes seed=42")
    expected = {"teacher": "assets/models/Qwen3.5-35B-A3B", "student": "assets/models/Qwen3-4B"}
    if 'reference' in cfg or set(cfg.get('tokenizer_paths',{}))!=set(MODEL_ROLES):
        raise ValueError('Current overlap uses teacher and student only; remove Ref configuration')
    if any(cfg.get(k) != v for k, v in expected.items()):
        raise ValueError("This campaign uses the selected Qwen3.5 teacher and Qwen3-4B student")
    if cfg.get("optimizer_updates") != 0:
        raise ValueError("The selected screen requires zero updates")
    if cfg.get('direction')!='T-S' or cfg.get('direction_version')!=DIRECTION_VERSION:
        raise ValueError('Select the current T-S direction protocol and a new run')
    if type(cfg.get('sv_max_context_len')) is not int or cfg['sv_max_context_len'] < 2:
        raise ValueError('Set sv_max_context_len >= 2 for the SV-training extraction protocol')
    from steer_opd.operators import validate_operator
    validate_operator(cfg.get('operator','slerp'),cfg['alpha'])
    if LINEAR in cfg.get('strength_policies',['fixed']):
        validate_operator('slerp',2*cfg['alpha'])
    layers=cfg['layers']
    blocks=read_json(model_path(cfg['student'])/'config.json')['num_hidden_layers']
    if len(layers)!=1 or type(layers[0]) is not int or not 0<=layers[0]<blocks:
        raise ValueError(f'Select one student layer in 0..{blocks-1}')
    if type(cfg.get('include_teacher_ablation',False)) is not bool:
        raise ValueError('include_teacher_ablation must be a boolean')
    if cfg.get('include_teacher_ablation'):
        text=read_json(model_path(cfg['teacher'])/'config.json')
        count=text.get('text_config',text)['num_hidden_layers']
        teacher_layers=cfg.get('teacher_layers',[])
        if (len(teacher_layers)!=1 or type(teacher_layers[0]) is not int
                or not 0<=teacher_layers[0]<count):
            raise ValueError(f'Select one teacher layer in 0..{count-1}')
        if cfg.get('teacher_direction_version')!=TEACHER_DIRECTION_VERSION:
            raise ValueError('Select the teacher-space S-T direction protocol')
    if cfg.get("metric") != METRIC or cfg.get("screen_backend") not in ("hf_batched_static", "easysteer"):
        raise ValueError("Explicitly select the aligned project metric and an available backend")
    if cfg.get("metric_version", METRIC_VERSION) != METRIC_VERSION or cfg.get("matching_rule_version", RULE_VERSION) != RULE_VERSION:
        raise ValueError("Old overlap definitions require new metrics, thresholds, masks and a new run")
    if cfg.get("returned_top_k", 16) != 16 or cfg.get("P_overlap_definition", P_DEFINITION) != P_DEFINITION:
        raise ValueError("Both current overlap metrics require raw Top16, without renormalization")
    if cfg.get("gate_metric", "overlap") not in ("overlap", "P_overlap") or cfg.get("rank_metric", "overlap") not in ("overlap", "P_overlap"):
        raise ValueError("Select one explicit static gate/ranking metric; both metrics are always recorded")
    if cfg.get('data_policy')=='all_train':
        if cfg.get('stop_after','panel')!='panel':raise ValueError('Unified training overlap requires the complete panel')
        if cfg['teacher_gpus'] not in (1,2) or any(c['gate']!='all' for c in scheme_conditions(cfg)):
            raise ValueError('The unified training pool requires all-position injection and one or two teacher GPUs')
    elif (not 0 < cfg["select_fraction"] < 1 or cfg["minimum_split_questions"] < 1
            or not 1 <= cfg["confirm_top_n"] <= len(scheme_conditions(cfg)) or not 0 < cfg["gate_target_rate"] < 1
            or cfg["teacher_gpus"] not in (1, 2)):
        raise ValueError("Invalid split/calibration/confirmation/GPU budget")


def scheme_conditions(cfg):
    conditions=[]
    policies=cfg.get('strength_policies',['fixed'])
    if cfg.get('phase')=='overlap' and policies!=['fixed']:
        raise ValueError('Overlap uses fixed strength only; annealing is deferred to training')
    if policies not in (['fixed'],['fixed',LINEAR]):raise ValueError('Use fixed or fixed plus linear_80_tail20')
    for policy in policies:
        for gate in ('all','static_overlap'):
            for granularity in ('global','per_question'):
                index=len(conditions)+1
                conditions.append({'id':f'scheme_{index:02d}','direction':'T-S',
                    'gate':gate,'granularity':granularity,'bank':f'T-S_{granularity}',
                    'operator':cfg.get('operator','slerp'),'alpha':cfg['alpha'],'layers':cfg['layers'],
                    'strength_policy':policy,'paired_condition_id':f'scheme_{index+4 if index<=4 else index-4:02d}',
                    'direction_version':DIRECTION_VERSION})
    if policies==['fixed']:
        for condition in conditions:condition.pop('paired_condition_id',None)
    for condition in conditions:condition['steer_role']='student'
    if cfg.get('include_teacher_ablation'):
        if policies!=['fixed']:
            raise ValueError('Teacher ablation currently supports fixed strength only')
        for gate in ('all','static_overlap'):
            for granularity in ('global','per_question'):
                conditions.append({'id':f'scheme_{len(conditions)+1:02d}', 'direction':'S-T',
                    'steer_role':'teacher','gate':gate,'granularity':granularity,
                    'bank':f'S-T_{granularity}','operator':cfg.get('operator','slerp'),'alpha':cfg['alpha'],
                    'layers':cfg['teacher_layers'],'strength_policy':'fixed',
                    'direction_version':TEACHER_DIRECTION_VERSION})
    selected=cfg.get('scheme_ids',[c['id'] for c in conditions])
    if (not selected or len(set(selected))!=len(selected)
            or set(selected)-{c['id'] for c in conditions}):
        raise ValueError(f'Select unique scheme IDs from scheme_01..scheme_{len(conditions):02d}')
    chosen=[c for c in conditions if c['id'] in selected]
    if cfg.get('operator','slerp')=='additive':
        if any(c['id']!='scheme_05' for c in chosen):
            raise ValueError('Raw additive search requires S-T/all/global scheme_05 only')
        for c in chosen:c['additive_scaling']='raw_mean_difference_v1'
    return chosen


def prepare_questions(cfg, codecs):
    original = read_json(project_path(cfg["question_source"]))
    if cfg.get('question_count') is not None and len(original)!=cfg['question_count']:
        raise ValueError('Question source differs from the declared experiment size')
    if len({r['id'] for r in original})!=len(original):raise ValueError('Duplicate question IDs')
    questions, rejected = [], {}
    for row in original:
        prompts = {role: prompt_ids(codec.tokenizer, row["question"], enable_thinking=cfg["enable_thinking"])
                   for role, codec in codecs.items()}
        if any(len(p) > cfg["max_prompt_tokens"] for p in prompts.values()):
            rejected[row["id"]] = "native_prompt_exceeds_frozen_limit"
            continue
        questions.append({"id": row["id"], "question": row["question"], "answer": row["answer"],
                          "student_prompt_ids": prompts["student"]})
    if cfg.get('data_policy')!='all_train' and len(questions) < 2*cfg["minimum_split_questions"]:
        raise ValueError("Too few shared prompt-valid training questions")
    if cfg.get('question_count') is not None and len(questions)!=cfg['question_count']:
        raise ValueError('Prompt filtering would reduce the declared experiment size; prepare valid inputs first')
    data = project_path(cfg["data_dir"])
    if cfg.get('data_policy')=='all_train':
        if not questions:raise ValueError('The training pool cannot be empty')
        frozen_json(data/'questions.json',questions)
        frozen_json(data/'train_ids.json',{'schema':'all_training_questions_v1',
            'train_ids':[q['id'] for q in questions],'seed':cfg['seed'],
            'source_sha256':file_sha256(project_path(cfg['question_source'])),
            'uses_test':False,'scope':'same_questions_for_sv_overlap_and_opd'})
        return questions
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
    from steer_opd.generation_policy import presence_penalty
    if role not in ('teacher','student'):
        raise ValueError('Only teacher reference text and student source generation are supported')
    output = str(artifact_dir(cfg,'student' if source else 'teacher'))
    requests = [{"id": q["id"], "key": f"{q['id']}_0" if source else q["id"], "question": q["question"],
                 "answer": q["answer"], "seed": cfg["seed"],
                 "prompt_token_ids": prompt_ids(codec.tokenizer, q["question"], enable_thinking=cfg["enable_thinking"])}
                for q in questions]
    job = {**execution_policy(cfg), "model": cfg[role], "output": output, "requests": requests, "vector": None,
        "max_tokens": cfg["source_max_tokens"] if source else cfg["reference_max_tokens"],
        "temperature": 1. if source else cfg["reference_temperature"], "top_p": 1. if source else cfg["reference_top_p"],
        "presence_penalty": presence_penalty(cfg),
        "seed": cfg["seed"], "support_size": codec.support_size, "gpu_memory_utilization": cfg["gpu_memory_utilization"],
        "logprobs": 16 if source else None, "head_dtype": "float32", "normalization": "native",
        "backend_version": 8, "metric_version": METRIC_VERSION, "matching_rule_version": RULE_VERSION,
        "decode_cuda_graph": cfg.get("decode_cuda_graph", True),
        "tensor_parallel_size": cfg["teacher_gpus"] if role == "teacher" else 1}
    job.update({k: cfg[k] for k in ("student", "teacher", "tokenizer_paths")})
    job.update(role=role, enable_prefix_caching=True)
    job.update(experiment=cfg['experiment'],phase=cfg['phase'])
    job["reproducibility_protocol"] = 1
    job["resume_granularity"] = "engine_call"
    job["bind_checkpoint_content"] = cfg.get("bind_checkpoint_content", False)
    if codec.valid_ids != list(range(codec.support_size)):
        if source:
            raise ValueError("Source native raw-Top16 profile requires contiguous student IDs")
        job["allowed_token_ids"] = codec.valid_ids
    if source:
        job.update(save_rollout_top_k=16, logprobs_mode="raw_logprobs")
    if role == "teacher" and cfg.get("teacher_text_reuse"):
        job["reuse_committed_texts"] = cfg["teacher_text_reuse"]
    if source and cfg.get("source_rollout_reuse"):
        job["reuse_committed_rollout"] = cfg["source_rollout_reuse"]
    return job



def teacher_text_contract(job):
    """Generation semantics, independent of overlap statistics and GPU receipts."""
    metadata = {"execution_hold", "execution_blockers", "gpu_acceptance_receipt", "gpu_acceptance_binding",
                "backend_version", "metric_version", "matching_rule_version", "model_content_id", "reuse_committed_texts"}
    if job.get("role") != "teacher" or job.get("vector") is not None or job.get("logprobs") is not None or job.get("save_rollout_top_k"):
        raise ValueError("Only unsteered committed teacher text can cross a metric-version migration")
    return {k: v for k, v in job.items() if k not in metadata}


def validate_teacher_text_reuse(job):
    """Keep the original generation manifest; verify every byte before reuse."""
    from steer_opd.checkpoint_identity import checkpoint_content_id
    from steer_opd.experiment_runtime import StageCheckpoint
    output = project_path(job["output"])
    receipt = read_json(project_path(job["reuse_committed_texts"]))
    if receipt.get("schema") != "teacher_text_reuse_v1":
        raise ValueError("Invalid teacher text reuse receipt")
    for name, field in (("complete.json", "complete_sha256"), ("job.json", "job_sha256")):
        if file_sha256(output/name) != receipt[field]:
            raise ValueError("Committed teacher generation identity changed")
    packet = read_json(output/"job.json"); old = packet["job"]
    if packet["hash"] != digest(old) or teacher_text_contract(old) != teacher_text_contract(job):
        raise ValueError("Teacher text generation settings or requests changed")
    if digest(teacher_text_contract(job)) != receipt["generation_contract_sha256"]:
        raise ValueError("Teacher text reuse contract changed")
    content = checkpoint_content_id(model_path(job["model"]))
    if old.get("model_content_id") != content or receipt["model_content_id"] != content:
        raise ValueError("Teacher checkpoint contents changed")
    runtime = read_json(output/"reproducibility.json")
    binding = digest({"job": old, "runtime": runtime["binding"]})
    if runtime["job_hash"] != digest(old) or not (output/"stage_input.json").is_file():
        raise ValueError("Original teacher generation provenance is incomplete")
    marker = StageCheckpoint(output, binding).load()
    expected = {"job.json", *(r["key"]+".json" for r in job["requests"])}
    if marker is None or marker.get("requests") != len(job["requests"]) or set(marker["files"]) != expected:
        raise ValueError("Teacher text batch is not completely committed")
    for request in job["requests"]:
        row = read_json(output/(request["key"]+".json"))
        if (any(row.get(k) != v for k, v in request.items()) or row.get("model") != job["model"]
                or row.get("model_content_id") != content or row.get("job_hash") != digest(old)
                or row.get("steering") is not False or not row.get("token_ids")):
            raise ValueError("Teacher response differs from its original request or checkpoint")
    return len(job["requests"])


def source_rollout_contract(job):
    """Preserve generation and raw-Top16 semantics across a recorded code fix."""
    if (job.get("role") != "student" or job.get("vector") is not None
            or job.get("logprobs") != 16 or job.get("save_rollout_top_k") != 16
            or job.get("logprobs_mode") != "raw_logprobs" or job.get("metric_version") != METRIC_VERSION
            or job.get("matching_rule_version") != RULE_VERSION):
        raise ValueError("Only the current unsteered raw-Top16 student rollout can be reused")
    metadata = {"execution_hold", "execution_blockers", "gpu_acceptance_receipt", "gpu_acceptance_binding",
                "model_content_id", "reuse_committed_rollout"}
    return {k: v for k, v in job.items() if k not in metadata}


def validate_source_rollout_reuse(job):
    """Verify original text/probability commits without rebinding or rewriting them."""
    from steer_opd.checkpoint_identity import checkpoint_content_id
    from steer_opd.experiment_runtime import StageCheckpoint, source_revision
    output = project_path(job["output"])
    receipt = read_json(project_path(job["reuse_committed_rollout"]))
    if (receipt.get("schema") != "source_rollout_reuse_v1"
            or receipt.get("validated_source_revision") != source_revision()[0]):
        raise ValueError("Student rollout reuse requires a receipt for the current code")
    for name, field in (("complete.json", "complete_sha256"), ("job.json", "job_sha256")):
        if file_sha256(output/name) != receipt[field]:
            raise ValueError("Committed student rollout identity changed")
    packet = read_json(output/"job.json"); old = packet["job"]
    if packet["hash"] != digest(old) or source_rollout_contract(old) != source_rollout_contract(job):
        raise ValueError("Student rollout settings or requests changed")
    if digest(source_rollout_contract(job)) != receipt["generation_contract_sha256"]:
        raise ValueError("Student rollout reuse contract changed")
    content = checkpoint_content_id(model_path(job["model"]))
    if old.get("model_content_id") != content or receipt["model_content_id"] != content:
        raise ValueError("Student checkpoint contents changed")
    runtime = read_json(output/"reproducibility.json")
    if (runtime["job_hash"] != digest(old)
            or runtime["source_revision"] != receipt["original_source_revision"]
            or not (output/"stage_input.json").is_file()):
        raise ValueError("Original student rollout provenance changed")
    marker = StageCheckpoint(output, digest({"job": old, "runtime": runtime["binding"]})).load()
    expected = {"job.json", *(r["key"]+".json" for r in job["requests"]),
                *("topk/"+r["key"]+".pt" for r in job["requests"])}
    if marker is None or marker.get("requests") != len(job["requests"]) or set(marker["files"]) != expected:
        raise ValueError("Student rollout batch is not completely committed")
    for request in job["requests"]:
        row = read_json(output/(request["key"]+".json"))
        if (any(row.get(k) != v for k, v in request.items()) or row.get("model") != job["model"]
                or row.get("model_content_id") != content or row.get("job_hash") != digest(old)
                or row.get("steering") is not False or row.get("token_record_schema") != 2
                or row.get("steering_mask") != [False]*len(row["token_ids"])
                or row.get("rollout_topk") != "topk/"+request["key"]+".pt"):
            raise ValueError("Student rollout differs from its original request or checkpoint")
        load_rollout_topk(output, row, 16, job["support_size"])
    return len(job["requests"])


def make_base_job(cfg, codecs, rows):
    from steer_opd.generation_policy import presence_penalty
    run = cfg["run_dir"]
    conditions = scheme_conditions(cfg)
    banks = {f'{direction}_{granularity}':str(artifact_dir(cfg,'bank_teacher' if direction=='S-T' else 'bank_student')/f'{direction}_{granularity}.pt')
             for direction in dict.fromkeys(c['direction'] for c in conditions)
             for granularity in ('global','per_question')}
    schedule=None
    if any(c['strength_policy']==LINEAR for c in conditions):
        expected=freeze_schedule(read_json(project_path(cfg['question_source'])),seed=cfg['seed'])
        saved=project_path(cfg['run_dir'])/'strength_schedule.json'
        schedule=validate_schedule(read_json(saved)) if saved.exists() else expected
        if schedule!=expected:raise ValueError('Frozen strength order differs from the original question pool')
        for condition in conditions:
            if condition['strength_policy']==LINEAR:condition['schedule_id']=schedule['schedule_id']
    job = {k: cfg[k] for k in ("student", "teacher", "enable_thinking", "tokenizer_paths")}
    job.update(**execution_policy(cfg), stage="static_overlap_screen", optimizer_updates=0, rows=rows,
        questions=f"{cfg['data_dir']}/questions.json", source_dir=str(artifact_dir(cfg,'student')), support_size=codecs["student"].support_size,
        teacher_scores=str(artifact_dir(cfg,'native_teacher')),
        alignment_dir=str(artifact_dir(cfg,'alignment')), source_metrics_dir=str(artifact_dir(cfg,'source_metrics')),
        banks=banks, conditions=conditions,
        identity={"metric": METRIC, "metric_version": METRIC_VERSION, "matching_rule_version": RULE_VERSION,
                  "P_overlap_definition": P_DEFINITION, "topk_tie_policy": TOPK_TIE_POLICY,
                  'direction_version':DIRECTION_VERSION,
                  'direction_versions':{c['direction']:DIRECTION_VERSIONS[c['direction']] for c in conditions}},
        backend=cfg["screen_backend"], batch_size=cfg["student_score_batch_size"],
        teacher_batch_size=cfg["teacher_score_batch_size"],
        gpu_memory_utilization=cfg["gpu_memory_utilization"], chunk=cfg["logit_chunk_size"])
    if cfg.get('candidate_tensor_compression'):
        job['candidate_tensor_compression']=cfg['candidate_tensor_compression']
    if cfg.get('candidate_tensor_format'):
        job['candidate_tensor_format']=cfg['candidate_tensor_format']
    job.update(require_dual_metrics=cfg.get("require_dual_metrics", True),
        seed=cfg["seed"], presence_penalty=presence_penalty(cfg), reproducibility_protocol=1, resume_granularity="engine_call",
        bind_checkpoint_content=cfg.get("bind_checkpoint_content", False),
        gate_metric=cfg.get("gate_metric", "overlap"), rank_metric=cfg.get("rank_metric", "overlap"),
        probability_backend=cfg.get("probability_backend", "easysteer"), teacher_gpus=cfg["teacher_gpus"])
    job.update(direction='T-S',direction_version=DIRECTION_VERSION,experiment=cfg['experiment'],phase=cfg['phase'])
    if schedule is not None:job['strength_schedule']=schedule
    return job


def role_screen_job(job, role):
    """Separate GPU processes; preserve a common student-coordinate metric/mask."""
    if role not in MODEL_ROLES:raise ValueError('Invalid intervention role')
    conditions=[c for c in job['conditions'] if c.get('steer_role','student')==role]
    if not conditions:return None
    mixed=any(c.get('steer_role','student')=='teacher' for c in job['conditions'])
    output=job['output']+'/'+role if mixed else job['output']
    result={**job,'conditions':conditions,'steer_role':role,'output':output,
            'direction':'T-S' if role=='student' else 'S-T',
            'direction_version':DIRECTION_VERSIONS['T-S' if role=='student' else 'S-T'],
            'banks':{c['bank']:job['banks'][c['bank']] for c in conditions}}
    if role=='teacher' and any(c.get('steer_role','student')=='student' for c in job['conditions']):
        result['paired_student_baseline']=job['output']+'/student/baseline'
    return result


def merge_role_summaries(job):
    """Merge summaries only; never mutate committed role-specific manifests."""
    result={}
    for role in MODEL_ROLES:
        part=role_screen_job(job,role)
        if part is None:continue
        folder=project_path(part['output'])
        if not (folder/'complete.json').is_file():raise ValueError('Role screen is not complete')
        saved=read_json(folder/'summaries.json')
        if set(saved)!={c['id'] for c in part['conditions']}:
            raise ValueError('Role screen candidate identities changed')
        result.update(saved)
    if any(c.get('steer_role','student')=='teacher' for c in job['conditions']):
        frozen_json(project_path(job['output'])/'summaries.json',result)
    return result


def prepare_initial_jobs(cfg, codecs):
    questions = prepare_questions(cfg, codecs)
    run = project_path(cfg["run_dir"])
    jobs=project_path(metadata_dir(cfg))/'jobs'
    if cfg.get('stop_after')=='source_overlap':
        frozen_json(jobs/'source.json',generation_job(cfg,codecs['student'],'student',questions,source=True))
        frozen_json(jobs/'alignment_and_scoring.json',make_source_overlap_job(cfg,codecs,f"{cfg['run_dir']}/source_rows.json"))
        return questions
    if LINEAR in cfg.get('strength_policies',[]):
        original=read_json(project_path(cfg['question_source']))
        frozen_json(run/'strength_schedule.json',freeze_schedule(original,seed=cfg['seed']))
    for role in ("teacher",):
        frozen_json(jobs/f"reference_{role}.json", generation_job(cfg, codecs[role], role, questions))
    frozen_json(jobs/'activations.json',cfg)
    frozen_json(jobs/'alignment_and_scoring.json', make_base_job(cfg, codecs, f"{cfg['run_dir']}/source_rows.json"))
    return questions


def make_source_overlap_job(cfg, codecs, rows):
    job=make_base_job(cfg,codecs,rows)
    for key in ('banks','conditions','strength_schedule'):
        job.pop(key,None)
    job.update(stage='source_overlap',stop_after='source_overlap')
    return job


def prepare_screen_jobs(cfg, base_job, codecs):
    """Calibrate only on frozen select questions, then freeze the selected T-S jobs."""
    rows, plans, role_rows, _ = prepare_plans(base_job, codecs)
    if cfg.get('allowed_question_ids') is not None:
        allowed=set(cfg['allowed_question_ids']);rows=[r for r in rows if r['id'] in allowed]
    split = read_json(project_path(cfg["data_dir"]) / "split.json")
    select_ids, confirm_ids = set(split["select_ids"]), set(split["confirm_ids"])
    selected = [r for r in rows if r["id"] in select_ids]
    confirmed = [r for r in rows if r["id"] in confirm_ids]
    if min(len({r["id"] for r in selected}), len({r["id"] for r in confirmed})) < cfg["minimum_split_questions"]:
        raise ValueError("Too few common usable questions in select/confirm; split is not silently changed")
    teachers = {}; teacher_hashes = {}
    native_rows = {role: {r["key"]: r for r in values} for role, values in role_rows.items()}
    for row in selected:
        key = safe_key(row)
        teachers[key] = torch.load(project_path(base_job["teacher_scores"]) / f"{key}.pt", weights_only=True, map_location="cpu")
        teacher_hashes[key] = file_sha256(project_path(base_job['teacher_scores'])/f'{key}.pt')
        for role, record in [("teacher", teachers[key])]:
            expected = native_score_binding(base_job, role, native_rows[role][key], codecs[role])
            if record.get("binding") != expected:
                raise ValueError("Native source scoring identity changed before threshold calibration")
            from steer_opd.static_masks import validate_native_score
            validate_native_score(record, len(native_rows[role][key]["token_ids"]), codecs[role].support_size)
    adapter = AlignedMetricAdapter(plans, codecs["student"], codecs["teacher"],
                                   require_dual=base_job.get("require_dual_metrics", False))
    gate_metric = cfg.get('gate_metric', 'overlap')
    names = ('overlap','P_overlap') if base_job.get('require_dual_metrics') else ('overlap',)
    values = {name:[] for name in names}
    for row in selected:
        if row.get("model") != base_job["student"]:
            raise ValueError("Threshold source came from a different student")
        score = load_rollout_topk(project_path(base_job["source_dir"]), row, 16, base_job["support_size"])
        key = safe_key(row)
        metric = cached_source_metrics(base_job, row, score, teachers[key], adapter,
            file_sha256(project_path(base_job['source_dir'])/row['rollout_topk']), teacher_hashes[key])
        for name in names:
            values[name].extend(metric[name][metric['alignment_mask']].double().tolist())
    calibrations = {name:calibrate_threshold(v,cfg['gate_target_rate'],'lt',lower_bound=0.)
                    for name,v in values.items()}
    calibration = calibrations[gate_metric]
    if not 0 <= calibration["threshold"] <= 1:
        raise ValueError("Calibrated overlap threshold outside the declared [0,1] domain")
    run = project_path(cfg["run_dir"])
    calibration_dir=project_path(artifact_dir(cfg,'calibration'))
    frozen_json(calibration_dir / "thresholds.json", {"metrics": calibrations, "gate_metric": cfg.get("gate_metric", "overlap"),
                                        "metric_version": METRIC_VERSION, "matching_rule_version": RULE_VERSION,
                                        "uses_confirm": False, "uses_test": False})
    frozen_json(calibration_dir / "threshold.json", {**calibration, "metric": cfg.get("gate_metric", "overlap"), "select_question_ids": sorted({r['id'] for r in selected}),
                                         "metric_version": METRIC_VERSION, "matching_rule_version": RULE_VERSION,
                                         "uses_confirm": False, "uses_test": False})
    frozen_json(run / "select_rows.json", selected)
    frozen_json(run / "confirm_rows.json", confirmed)
    select_job = {**base_job, "rows": f"{cfg['run_dir']}/select_rows.json", "alignment_dir": f"{cfg['run_dir']}/select_alignment",
                  "threshold": calibration["threshold"], "output": f"{cfg['run_dir']}/select"}
    # Native scoring rows depend on question/native tokens, not which split
    # later consumes them; alignment IDs are preserved between these plans.
    confirm_job = {**base_job, "rows": f"{cfg['run_dir']}/confirm_rows.json", "alignment_dir": f"{cfg['run_dir']}/confirm_alignment",
                   "threshold": calibration["threshold"], "output": f"{cfg['run_dir']}/confirm"}
    if cfg.get('layout_version')==1:
        select_job['alignment_source_dir']=str(artifact_dir(cfg,'alignment'))
        confirm_job['alignment_source_dir']=str(artifact_dir(cfg,'alignment'))
    frozen_json(project_path(metadata_dir(cfg))/'jobs/select.json', select_job)
    return select_job, confirm_job


def prepare_all_training_job(cfg,base_job,codecs):
    """Score every training question once; all injection needs no calibrated gate."""
    rows,_,_,_=prepare_plans(base_job,codecs)
    expected=read_json(project_path(cfg['data_dir'])/'train_ids.json')['train_ids']
    if {r['id'] for r in rows}!=set(expected):
        raise ValueError('Overlap requests must cover the complete training question pool')
    if any(c['gate']!='all' for c in base_job['conditions']):
        raise ValueError('Unified training overlap currently requires all-position injection')
    run=project_path(cfg['run_dir'])
    frozen_json(run/'train_rows.json',rows)
    job={**base_job,'rows':str((run/'train_rows.json').relative_to(ROOT)),
         'alignment_dir':str((run/'train_alignment').relative_to(ROOT)),
         'alignment_source_dir':str(artifact_dir(cfg,'alignment')),
         'output':str((run/'train').relative_to(ROOT)),
         'threshold':0.,'data_policy':'all_train'}
    frozen_json(project_path(metadata_dir(cfg))/'jobs/train.json',job)
    return job
