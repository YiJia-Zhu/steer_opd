#!/usr/bin/env python3
"""Prepare frozen masks on CPU, then optionally score all conditions, never train.

Inputs are unsteered source rows, raw student rollout Top16, teacher caches,
explicit metric identities and two S0 T-S banks. Native IDs require the same
mapping. The separate Compass-style adapter uses aligned byte continuations;
it never labels its result as native cross-tokenizer token-ID overlap.
"""

import argparse
import os
from pathlib import Path

import torch

from steer_opd.direction_bank import file_sha256
from steer_opd.overlap_tuning import paired_summary, position_metrics
from steer_opd.resumable_core import ROOT, atomic_json, digest, read_json
from steer_opd.static_masks import freeze_static_mask, native_overlap, validate_native_score
from steer_opd.static_screen import candidate_vectors, score_native_candidate
from steer_opd.method_config import Method
from steer_opd.execution_policy import require_default_scheduler_options, require_execution_allowed
from steer_opd.experiment_runtime import initialize_stage, Progress
from steer_opd.tensor_checkpoint import StageCheckpoint
from steer_opd.strength_schedule import row_strength, validate_schedule


def prepare_engine_requests(rows, masks, bank, condition, schedule=None):
    """Prepare every request before one engine call; never group calls by alpha."""
    from steer_opd.easysteer_static import build_spec
    if schedule is not None:validate_schedule(schedule)
    specs,metadata,vector_cache=[],[],{}
    for row in rows:
        key=row.get('key',row['id']);strength=row_strength(condition,row,schedule)
        alpha=strength['effective_alpha']
        requested=(masks[key]['requested_mask'] if condition and condition['gate']=='static_overlap'
                   else [bool(condition)]*len(row['token_ids']))
        vector_id=None;spec=False;nonzero=False
        if condition and alpha!=0:
            assigned=row['id'] if condition['granularity']=='per_question' else 'global'
            if assigned not in vector_cache:vector_cache[assigned]=candidate_vectors(bank,row,condition)
            vectors,base_id=vector_cache[assigned]
            vector_id=digest({'base_vector_id':base_id,'strength':strength})
            nonzero=any(bool(torch.count_nonzero(v)) for v in vectors.values())
            actual=[bool(g and nonzero) for g in requested]
            spec=build_spec(row,actual,vectors,operator=condition.get('operator','slerp'),alpha=alpha)
        else:actual=[False]*len(requested)
        specs.append(spec)
        metadata.append({'mask_id':masks[key]['mask_id'],'source_binding':masks[key]['source_binding'],
            'binding':digest({'mask_id':masks[key]['mask_id'],'condition':condition,'vector_id':vector_id,'strength':strength,'row':row}),
            'vector_id':vector_id,'condition':condition,'requested_mask':torch.tensor(requested),
            'steering_mask':torch.tensor(actual),'relative_perturbation_available':False,
            'zero_vector_no_intervention':bool(condition is not None and alpha!=0 and not nonzero),
            'zero_strength_no_intervention':bool(condition is not None and alpha==0),
            'direction_lookup_skipped':bool(condition is not None and alpha==0),**strength})
    return specs,metadata


def path(value):
    result = Path(value)
    result = result if result.is_absolute() else ROOT / result
    if not result.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("Static screen inputs/outputs must stay within the project")
    return result


def prepare(job):
    if job.get('strength_schedule') is not None:
        validate_schedule(job['strength_schedule'])
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
            strength=row_strength(condition,row,job.get('strength_schedule'))
            if strength['effective_alpha']!=0:candidate_vectors(bank,row,{**condition,'alpha':strength['effective_alpha']})
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
    if job.get('steer_role')=='teacher':
        from steer_opd.teacher_screen import run_teacher_screen
        return run_teacher_screen(job,rows,masks,teachers,banks)
    require_execution_allowed(job)
    require_default_scheduler_options(job)
    runtime=initialize_stage(job,path(job["output"]))
    with Progress(path(job["output"]),"fixed_prefix_screen",len(rows)*(1+len(job["conditions"])),unit="score",show_bar=False) as progress:
        return _run(job,rows,masks,teachers,banks,runtime,progress)


def _run(job, rows, masks, teachers, banks, runtime, progress):
    from steer_opd.resumable_core import load_model, require_gpu
    from steer_opd.static_plotting import plot_comparisons, plot_dual_comparisons
    from steer_opd.compass_job import model_path
    os.environ["STEER_OPD_SUPPORT_SIZE"] = str(job["support_size"])
    output = path(job["output"])
    schedule=job.get('strength_schedule')
    if schedule is not None:validate_schedule(schedule)
    assignments={c['id']:{r.get('key',r['id']):row_strength(c,r,schedule) for r in rows} for c in job['conditions']}
    assignments_path=output/'strength_assignments.json'
    if assignments_path.exists() and read_json(assignments_path)!=assignments:
        raise ValueError('Frozen candidate strength assignments changed')
    atomic_json(assignments_path,assignments)
    metric_adapter = getattr(teachers, "metric_adapter", None)
    plot_keys = ([r.get("key", r["id"]) for r in read_json(path(job["rows"]))[:3]]
                 if metric_adapter is not None else None)
    phase_binding = digest({"manifest":read_json(output / "manifest.json"),"runtime":runtime})
    stores = {}
    def plots(condition, base, records):
        plot = plot_dual_comparisons if job.get("require_dual_metrics") else plot_comparisons
        plot(rows, base, records, output / condition["id"] / "comparison", requested_keys=plot_keys)
    def cached(name):
        stores[name]=StageCheckpoint(output/name,digest({'candidate':name,'binding':phase_binding}),
            tensor_format=job.get('candidate_tensor_format',job.get('candidate_tensor_compression')))
        return stores[name].load_tensors([r.get("key",r["id"]) for r in rows])
    def commit(name,records):
        if ready[name] is None:
            with progress.phase_scope('save_complete_candidate',condition=name):
                stores[name].save_tensors(records)
            ready[name]=records
            progress.mark_committed(progress.committed+len(rows))
    required = ["baseline", *[c["id"] for c in job["conditions"]]]
    ready = {name: cached(name) for name in required}
    progress.resume(sum(len(v) for v in ready.values() if v is not None))
    if all(ready[name] is not None for name in required) and (output / "complete.json").exists():
        # Rebuild requested CPU plots from cached data if necessary.
        for condition in job["conditions"]:
            plots(condition, ready["baseline"], ready[condition["id"]])
        save_policy_comparisons(output,job,rows,ready)
        print("All static conditions already complete; reused committed probability records")
        return
    needs_model = any(ready[name] is None for name in required)
    backend = job.get("backend", "hf_batched_static")
    if backend not in ("hf_batched_static", "easysteer"):
        raise ValueError("Unknown static backend; no automatic backend substitution")
    if needs_model:
        from steer_opd.experiment_runtime import compute_required
        compute_required()
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
            model = load_text_scorer(model_path(job["student"]), metric_adapter.student_codec, metric_adapter.teacher_codec)
    elif needs_model:
        from steer_opd.easysteer_static import require_easysteer
        require_easysteer()
        if metric_adapter is not None and metric_adapter.student_codec.valid_ids != list(range(job["support_size"])):
            raise ValueError("EasySteer support cropping requires a contiguous defined student vocabulary")
        os.environ["STEER_OPD_STATIC_SUPPORT"] = str(job["support_size"])
        from steer_opd.easysteer_static import create_engine
        model = create_engine(model_path(job["student"]), job["support_size"],
            max_model_len=max(len(r["prompt_token_ids"])+len(r["token_ids"]) for r in rows)+1,
            gpu_memory_utilization=job.get("gpu_memory_utilization", .75), seed=42, progress=progress,batch_invariant=True)
    def evaluate(condition):
        name=condition["id"] if condition else "baseline"
        if ready[name] is not None:return ready[name]
        start=progress.done
        bank=banks[condition["bank"]] if condition else next(iter(banks.values()))
        if backend=="hf_batched_static":
            records=score_native_candidate(model,rows,teachers,masks,bank,condition,
                baseline=baseline if condition else None,batch_size=job.get("batch_size",4),
                chunk=job.get("chunk",128),metric_adapter=metric_adapter,strength_schedule=job.get('strength_schedule'))
        else:
            records=evaluate_engine(condition,bank)
        progress.finished_call(start+len(rows))
        # All distributions are available: derive metrics/deltas after the engine call.
        for key,record in records.items():
            if condition:
                for metric in (("overlap","P_overlap") if job.get("require_dual_metrics") else ("overlap",)):
                    record[f"delta_{metric}"]=record[metric]-baseline[key][metric]
            else:
                gate=job.get("gate_metric","overlap")
                for metric,values in masks[key].get("source_metrics",{gate:masks[key]["source_overlap"]}).items():
                    record[f"source_rollout_vs_replay_delta_{metric}"]=record[metric].double()-torch.tensor(
                        [float("nan") if v is None else v for v in values],dtype=torch.float64)
                record["source_rollout_vs_replay_delta"]=record[f"source_rollout_vs_replay_delta_{gate}"]
        return records
    def evaluate_engine(condition,bank):
        from steer_opd.easysteer_static import fixed_prefix_scores
        specs,meta=prepare_engine_requests(rows,masks,bank,condition,job.get('strength_schedule'))
        with progress.phase_scope('continuous_candidate_scoring',condition=None if condition is None else condition['id']):
            scores=fixed_prefix_scores(model,rows,specs,include_sampled=metric_adapter is not None,
                progress=progress)
        with progress.phase_scope('compute_candidate_metrics'):
            return {r.get("key",r["id"]):{**score,
                **(position_metrics(score,teachers[r.get("key",r["id"])]) if metric_adapter is None else
                   metric_adapter(r.get("key",r["id"]),score,teachers[r.get("key",r["id"])])),**metadata}
                for r,score,metadata in zip(rows,scores,meta)}
    baseline=evaluate(None)
    commit("baseline",baseline)
    summaries = {}
    for condition in job["conditions"]:
        records = evaluate(condition)
        # Aggregate multiple responses by question before paired_summary.
        def by_question(values, metric="overlap"):
            groups = {}
            for row in rows:
                groups.setdefault(row["id"], []).append(values[row.get("key", row["id"])])
            return {qid: {name: torch.cat([v[metric if name == "overlap" else name][v.get("alignment_mask", torch.ones_like(v[metric], dtype=torch.bool))] for v in items]) for name in
                          ("overlap", "student_shared_mass", "teacher_shared_mass")} for qid, items in groups.items()}
        metric_names = ("overlap", "P_overlap") if job.get("require_dual_metrics") else ("overlap",)
        metric_summaries = {}
        for name in metric_names:
            summary = paired_summary(by_question(records, name), by_question(baseline, name))
            if name == "P_overlap":
                summary["token_weighted_value"] = summary.pop("token_weighted_overlap16")
                summary.pop("student_shared_mass"); summary.pop("teacher_shared_mass")
            summary["metric_name"] = name
            summary["question_weighted_value"] = sum(float(v["overlap"].mean()) for v in by_question(records, name).values()) / summary["questions"]
            total = sum(len(row["token_ids"]) for row in rows)
            summary.update(source_prediction_positions=total, valid_position_fraction=summary["positions"] / total,
                valid_span_count=sum(record.get("valid_span_count", 0) for record in records.values()),
                total_span_count=sum(record.get("total_span_count", 0) for record in records.values()),
                actual_steering_rate=sum(int(record["steering_mask"].sum()) for record in records.values()) / total)
            metric_summaries[name] = summary
            for row in rows:
                key = row.get("key", row["id"])
                records[key][f"delta_{name}"] = records[key][name]-baseline[key][name]
        rank_metric = job.get("rank_metric", "overlap")
        summaries[condition["id"]] = {"condition": condition, "metric": job["identity"]["metric"],
            'intervention_role':'student','metric_position_role':'student',
            **{k: next(iter(records.values()))[k] for k in ("metric_version", "matching_rule_version", "P_overlap_definition")
               if k in next(iter(records.values()))},
            'strength_assignment':{'scope':'zero_update_question_assignment',
                'policy':condition.get('strength_policy','fixed'),'schedule_id':schedule['schedule_id'] if schedule else None,
                'question_count':len({r['id'] for r in rows}),
                'zero_strength_requests':sum(v['effective_alpha']==0 for v in assignments[condition['id']].values()),
                'mean_alpha':sum(v['effective_alpha'] for v in assignments[condition['id']].values())/len(rows),
                'min_alpha':min(v['effective_alpha'] for v in assignments[condition['id']].values()),
                'max_alpha':max(v['effective_alpha'] for v in assignments[condition['id']].values())},
            "ranking_metric": rank_metric, "metrics": metric_summaries,
            **metric_summaries[rank_metric]}
        atomic_json(output / "summaries.json", summaries)
        # Deltas are derived from the committed baseline; persist both readouts.
        commit(condition["id"], records)
        with progress.phase_scope("plotting",condition=condition["id"]):
            plots(condition, baseline, records)
    save_policy_comparisons(output,job,rows,ready)
    atomic_json(output / "complete.json", {"status": "awaiting_user_review", "parameter_updates": 0,
                "candidates": len(summaries), "backend": backend, "automatic_opd": False})


def save_policy_comparisons(output,job,rows,records):
    """CPU-only fixed/annealed paired differences, including the zero-strength tail."""
    comparisons={}
    for condition in job['conditions']:
        if condition.get('strength_policy')!='linear_80_tail20':continue
        peer=condition.get('paired_condition_id')
        if peer not in records or records[peer] is None:continue
        annealed=records[condition['id']];fixed=records[peer]
        groups={'all':rows,'positive_schedule':[r for r in rows if annealed[r.get('key',r['id'])]['effective_alpha']>0],
                'zero_strength_tail':[r for r in rows if annealed[r.get('key',r['id'])]['effective_alpha']==0]}
        result={}
        for group,subset in groups.items():
            result[group]={}
            for metric in ('overlap','P_overlap') if job.get('require_dual_metrics') else ('overlap',):
                pooled=[{},{}]
                for row in subset:
                    key=row.get('key',row['id']);left,right=annealed[key],fixed[key]
                    valid=torch.isfinite(left[metric])&torch.isfinite(right[metric])
                    if not valid.any():continue
                    for target,value in zip(pooled,(left,right)):
                        target.setdefault(row['id'],[]).append(value[metric][valid])
                if not pooled[0]:continue
                def wrap(values):
                    return {qid:{'overlap':torch.cat(parts),'student_shared_mass':torch.zeros(sum(len(p) for p in parts)),
                                 'teacher_shared_mass':torch.zeros(sum(len(p) for p in parts))} for qid,parts in values.items()}
                summary=paired_summary(wrap(pooled[0]),wrap(pooled[1]))
                for name in ('student_shared_mass','teacher_shared_mass'):summary.pop(name,None)
                summary['metric_name']=metric
                summary['token_weighted_value']=summary.pop('token_weighted_overlap16')
                result[group][metric]=summary
        comparisons[condition['id']]={'fixed_id':peer,'annealed_id':condition['id'],
            'delta_definition':'annealed_minus_fixed','scope':'instantaneous_zero_update_same_prefix','groups':result}
    atomic_json(output/'strength_comparisons.json',comparisons)


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
