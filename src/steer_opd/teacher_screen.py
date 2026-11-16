"""Teacher-space S-T ablation on frozen student text; no student intervention."""
import torch

from steer_opd.compass_job import model_path, project_path, safe_key
from steer_opd.direction_bank import file_sha256
from steer_opd.execution_policy import require_execution_allowed, require_default_scheduler_options
from steer_opd.experiment_runtime import StageCheckpoint, Progress, initialize_stage, compute_required
from steer_opd.overlap_tuning import paired_summary
from steer_opd.resumable_core import atomic_json, digest, read_json
from steer_opd.static_masks import validate_native_score
from steer_opd.static_screen import candidate_vectors


def project_teacher_mask(plan, student_mask, teacher_length):
    """Use the frozen common spans, never student token indices in teacher space."""
    if len(student_mask)!=len(plan['alignment_mask']):
        raise ValueError('Student mask length differs from the source alignment')
    gates=[False]*teacher_length
    for unit in plan['units']:
        if not unit['valid']:continue
        a,b=unit['ranges']['student'];c,d=unit['ranges']['teacher']
        if not (0<=a<b<=len(student_mask) and 0<=c<d<=teacher_length):
            raise ValueError('Alignment span is outside the native prediction range')
        values=student_mask[a:b]
        if any(values) and not all(values):
            raise ValueError('A frozen source gate must be constant across a shared span')
        gates[c:d]=[bool(values[0])]*(d-c)
    return gates


def teacher_requests(rows, masks, native_rows, plans, bank, condition, build_spec=None):
    """One native teacher request per source; metrics remain on student positions."""
    if build_spec is None:
        from steer_opd.easysteer_static import build_spec
    specs=[];metadata=[];vectors=[];gates=[];cache={}
    for row in rows:
        key=safe_key(row);native=native_rows[key]
        alpha=condition['alpha'] if condition else 0.
        source_requested=(masks[key]['requested_mask'] if condition and condition['gate']=='static_overlap'
                          else [bool(condition)]*len(row['token_ids']))
        native_requested=(project_teacher_mask(plans[key],source_requested,len(native['token_ids']))
                          if condition and condition['gate']=='static_overlap'
                          else [bool(condition)]*len(native['token_ids']))
        values=None;vector_id=None;nonzero=False
        if condition and alpha!=0:
            assigned=row['id'] if condition['granularity']=='per_question' else 'global'
            if assigned not in cache:cache[assigned]=candidate_vectors(bank,row,condition)
            values,vector_id=cache[assigned]
            nonzero=any(bool(torch.count_nonzero(v)) for v in values.values())
        enabled=bool(nonzero and alpha!=0)
        native_actual=[bool(g and enabled) for g in native_requested]
        mapped_requested=[False]*len(row['token_ids'])
        mapped_actual=[False]*len(row['token_ids'])
        for unit in plans[key]['units']:
            if not unit['valid']:continue
            a,b=unit['ranges']['student'];c,d=unit['ranges']['teacher']
            mapped_requested[a:b]=[any(native_requested[c:d])]*(b-a)
            mapped_actual[a:b]=[any(native_actual[c:d])]*(b-a)
        specs.append(build_spec(native,native_actual,values,alpha=alpha) if enabled else False)
        vectors.append(values);gates.append(native_actual)
        meta={'mask_id':masks[key]['mask_id'],'source_binding':masks[key]['source_binding'],
            'condition':condition,'vector_id':vector_id,'intervention_role':'teacher',
            'metric_position_role':'student','steering_mask_semantics':'teacher_span_mapped_to_student',
            'requested_mask':torch.tensor(mapped_requested,dtype=torch.bool),
            'steering_mask':torch.tensor(mapped_actual,dtype=torch.bool),
            'native_requested_mask':torch.tensor(native_requested,dtype=torch.bool),
            'native_steering_mask':torch.tensor(native_actual,dtype=torch.bool),
            'native_prediction_positions':len(native['token_ids']),
            'relative_perturbation_available':False,'strength_policy':'fixed','schedule_index':None,
            'effective_alpha':alpha,'base_alpha':alpha,'schedule_id':None,
            'zero_strength_no_intervention':bool(condition and alpha==0),
            'zero_vector_no_intervention':bool(condition and alpha!=0 and not nonzero),
            'direction_lookup_skipped':bool(condition and alpha==0)}
        meta['binding']=digest({'source_mask':masks[key]['mask_id'],'native_row':native,
            'condition':condition,'vector_id':vector_id,'native_gate':native_actual})
        metadata.append(meta)
    return specs,metadata,vectors,gates


def summarize_teacher(rows, records, baseline):
    result={}
    for metric in ('overlap','P_overlap'):
        def group(values):
            groups={}
            for row in rows:
                record=values[safe_key(row)];valid=record['alignment_mask']
                item=groups.setdefault(row['id'],{'overlap':[],'student_shared_mass':[],'teacher_shared_mass':[]})
                for name in item:item[name].append(record[metric if name=='overlap' else name][valid])
            return {qid:{name:torch.cat(parts) for name,parts in fields.items()} for qid,fields in groups.items()}
        current=group(records);summary=paired_summary(current,group(baseline))
        if metric=='P_overlap':
            summary['token_weighted_value']=summary.pop('token_weighted_overlap16')
            summary.pop('student_shared_mass');summary.pop('teacher_shared_mass')
        total=sum(len(r['token_ids']) for r in rows)
        native_total=sum(v['native_prediction_positions'] for v in records.values())
        summary.update(metric_name=metric,
            question_weighted_value=sum(float(v['overlap'].mean()) for v in current.values())/len(current),
            source_prediction_positions=total,valid_position_fraction=summary['positions']/total,
            injection_prediction_positions=native_total,
            actual_steering_rate=sum(int(v['native_steering_mask'].sum()) for v in records.values())/native_total,
            mapped_student_steering_rate=sum(int(v['steering_mask'].sum()) for v in records.values())/total,
            valid_span_count=sum(v.get('valid_span_count',0) for v in records.values()),
            total_span_count=sum(v.get('total_span_count',0) for v in records.values()),
            intervention_role='teacher',metric_position_role='student',
            interpretation='teacher_adaptation_on_fixed_student_prefix_not_student_learning')
        result[metric]=summary
    return result


def paired_student_scores(job, rows, masks, teachers):
    """Use the exact committed student-arm baseline, when both arms are compared."""
    location=job.get('paired_student_baseline')
    if not location:return teachers.source_scores,{'kind':'frozen_source_student_top16'}
    folder=project_path(location);marker=read_json(folder/'complete.json')
    legacy=job.get('paired_student_baseline_adapter')
    if legacy:
        from steer_opd.paired_baseline import ADAPTER,equivalent_source_mask
        from steer_opd.artifact_reuse import StageArtifacts
        if legacy!=ADAPTER:raise ValueError('Unknown paired baseline adapter')
        owner,packet=StageArtifacts(job).validate(project_path(job['paired_student_baseline_descriptor']))
        if owner!=folder or packet['artifact_id']!=job['paired_student_baseline_artifact_id']:
            raise ValueError('Paired baseline descriptor identity changed')
        if not set(safe_key(r) for r in rows).issubset(marker['record_keys']):
            raise ValueError('The historical baseline is missing current source rows')
        saved={safe_key(r):torch.load(folder/(safe_key(r)+'.pt'),weights_only=True,map_location='cpu') for r in rows}
        manifest=read_json(folder.parent/'manifest.json')
    else:
        saved=StageCheckpoint(folder,marker['binding']).load_tensors([safe_key(r) for r in rows])
    if saved is None:raise ValueError('The paired student baseline is not committed')
    for row in rows:
        key=safe_key(row);record=saved[key]
        expected=masks[key]
        if legacy:
            expected=read_json(folder.parent/'masks'/f'{key}.json')
            if manifest['masks'][key]!=expected['mask_id']:raise ValueError('Original baseline mask manifest changed')
            equivalent_source_mask(expected,masks[key],row)
        if (record.get('condition') is not None or record['mask_id']!=expected['mask_id']
                or record['source_binding']!=expected['source_binding'] or record['steering_mask'].any()):
            raise ValueError('The paired student baseline prefix/mask identity changed')
        validate_native_score(record,len(row['token_ids']),job['support_size'])
    return saved,{'kind':'paired_student_arm_baseline','complete_sha256':file_sha256(folder/'complete.json'),
                  **({'mask_adapter':legacy} if legacy else {})}


def run_teacher_screen(job, rows, masks, teachers, banks):
    require_execution_allowed(job);require_default_scheduler_options(job)
    if job.get('steer_role')!='teacher' or any(c['direction']!='S-T' for c in job['conditions']):
        raise ValueError('Expected teacher-space S-T intervention conditions')
    output=project_path(job['output']);runtime=initialize_stage(job,output)
    student,fixed_identity=paired_student_scores(job,rows,masks,teachers)
    adapter=teachers.metric_adapter;native_rows=teachers.native_rows
    support=adapter.teacher_codec.support_size
    binding=digest({'manifest':read_json(output/'manifest.json'),'runtime':runtime,
                    'paired_student':fixed_identity,'protocol':'teacher_st_fixed_student_v1'})
    names=['baseline','teacher_unsteered_replay',*[c['id'] for c in job['conditions']]]
    if len(set(names))!=len(names):raise ValueError('Teacher candidate IDs overlap reserved baseline/replay names')
    stores={name:StageCheckpoint(output/name,digest({'phase':binding,'candidate':name})) for name in names}
    ready={name:store.load_tensors([safe_key(r) for r in rows]) for name,store in stores.items()}
    with Progress(output,'teacher_fixed_prefix_screen',len(rows)*len(names),unit='score',show_bar=False) as progress:
        progress.resume(sum(len(v) for v in ready.values() if v is not None))
        def records(scores, meta):
            result={}
            for row,score,metadata in zip(rows,scores,meta):
                key=safe_key(row);validate_native_score(score,len(native_rows[key]['token_ids']),support)
                metric=adapter(key,student[key],score)
                result[key]={**score,**metric,**metadata,'native_score_role':'teacher',
                    'fixed_student_identity':fixed_identity,'parameter_updates':0}
            return result
        def commit(name, values, *, completed=None, reused=False):
            stores[name].save_tensors(values);ready[name]=values
            if reused:
                progress.resume(progress.done+len(rows))
            else:
                progress.finished_call(completed)
                progress.mark_committed(progress.committed+len(rows))
        if ready['baseline'] is None:
            _,meta,_,_=teacher_requests(rows,masks,native_rows,adapter.plans,None,None)
            values=records([teachers[safe_key(r)] for r in rows],meta)
            if job.get('paired_student_baseline'):
                for key,value in values.items():
                    for metric in ('overlap','P_overlap'):
                        torch.testing.assert_close(value[metric],student[key][metric],rtol=0,atol=0,equal_nan=True)
            commit('baseline',values,reused=True)
        baseline=ready['baseline']
        model=None
        if any(ready[name] is None for name in names[1:]):
            compute_required()
            if job.get('backend')=='easysteer':
                from steer_opd.easysteer_static import require_easysteer_gpu,create_engine
                require_easysteer_gpu(job.get('teacher_gpus',2))
                native=list(native_rows.values())
                model=create_engine(model_path(job['teacher']),support,
                    max_model_len=max(len(r['prompt_token_ids'])+len(r['token_ids']) for r in native)+1,
                    tensor_parallel_size=job.get('teacher_gpus',2),batch_invariant=False,
                    gpu_memory_utilization=job.get('gpu_memory_utilization',.75),seed=42,progress=progress)
            elif job.get('backend')=='hf_batched_static':
                from steer_opd.compass_scoring import load_text_scorer
                model=load_text_scorer(model_path(job['teacher']),adapter.teacher_codec)
            else:raise ValueError('Unknown explicit teacher scoring backend')
        def evaluate(condition):
            name=condition['id'] if condition else 'teacher_unsteered_replay'
            if ready[name] is not None:return ready[name]
            start=progress.done
            bank=banks[condition['bank']] if condition else None
            # HF reference does not need an EasySteer installation for authoring.
            dummy=(lambda *args,**kw:False) if job['backend']=='hf_batched_static' else None
            specs,meta,vectors,gates=teacher_requests(rows,masks,native_rows,adapter.plans,bank,condition,build_spec=dummy)
            native=[native_rows[safe_key(r)] for r in rows]
            with progress.phase_scope('continuous_teacher_scoring',condition=name):
                if job['backend']=='easysteer':
                    from steer_opd.easysteer_static import fixed_prefix_scores
                    scores=fixed_prefix_scores(model,native,specs,include_sampled=True,progress=progress)
                else:
                    from steer_opd.static_screen import score_batch
                    scores=[]
                    size=job.get('teacher_batch_size',1)
                    for begin in range(0,len(rows),size):
                        scores.extend(score_batch(model,native[begin:begin+size],
                            vectors=vectors[begin:begin+size] if condition and condition['alpha']!=0 else None,
                            gates=gates[begin:begin+size] if condition and condition['alpha']!=0 else None,
                            alpha=condition['alpha'] if condition else 0.))
            with progress.phase_scope('compute_teacher_metrics',condition=name):
                values=records(scores,meta)
                for key,value in values.items():
                    for metric in ('overlap','P_overlap'):value['delta_'+metric]=value[metric]-baseline[key][metric]
            commit(name,values,completed=start+len(rows))
            return values
        replay=evaluate(None)
        drift=summarize_teacher(rows,replay,baseline)
        atomic_json(output/'teacher_repeat_drift.json',{'metrics':drift,'batch_invariant':False,
            'reference':'frozen_common_baseline','automatic_drift_subtraction':False})
        summaries={}
        from steer_opd.static_plotting import plot_dual_comparisons
        for condition in job['conditions']:
            values=evaluate(condition);metrics=summarize_teacher(rows,values,baseline)
            rank=job.get('rank_metric','overlap')
            summaries[condition['id']]={'condition':condition,'metrics':metrics,'ranking_metric':rank,
                'metric':job['identity']['metric'],'metric_version':4,
                'intervention_role':'teacher','common_baseline':fixed_identity,
                'teacher_repeat_drift':{m:drift[m]['question_weighted_delta_pp'] for m in drift},
                **metrics[rank]}
            atomic_json(output/'summaries.json',summaries)
            with progress.phase_scope('plotting',condition=condition['id']):
                plot_dual_comparisons(rows,baseline,values,output/condition['id']/'comparison',
                    requested_keys=[safe_key(r) for r in read_json(project_path(job['rows']))[:3]])
        atomic_json(output/'complete.json',{'status':'awaiting_user_review','candidates':len(summaries),
            'parameter_updates':0,'automatic_opd':False,'intervention_role':'teacher',
            'teacher_drift_recorded':True})
