"""GPU stage for isolated direction exploration; frozen teacher/student prefixes."""
from common import *
import argparse,json,time,torch
from directions import native_strengths
from steer_opd.compass_job import load_codecs,model_path,safe_key
from steer_opd.compass_scoring import AlignedMetricAdapter
from steer_opd.cpu_metrics import MetricExecutor
from steer_opd.experiment_runtime import StageCheckpoint,Progress,timestamp,configure_seed
from steer_opd.easysteer_static import create_engine,fixed_prefix_scores,build_spec
from steer_opd.static_masks import validate_native_score
from steer_opd.teacher_screen import summarize_teacher
from steer_opd.static_plotting import plot_dual_comparisons


def select_ids(config,phase):
 if phase=='pilot':return config['pilot_select_ids']
 return config['original_split'][phase]


def load_data(config,phase):
 split='select' if phase=='pilot' else phase
 job=read_json(ST/f'metadata/jobs/{split}_teacher.json')
 ids=set(select_ids(config,phase));rows=[r for r in read_json(input_path(job['rows'])) if r['id'] in ids]
 alignment=input_path(job.get('alignment_source_dir',job['alignment_dir']))
 teacher_lookup={r['key']:r for r in read_json(alignment/'teacher_rows.json')}
 native=[teacher_lookup[safe_key(r)] for r in rows]
 plans={safe_key(r):read_json(alignment/f'{safe_key(r)}.json') for r in rows}
 baseline={safe_key(r):load_packet(ST/split/'teacher/baseline'/f'{safe_key(r)}.pt') for r in rows}
 # Use the exact committed student replay paired with every existing teacher candidate.
 paired=input_path(job['paired_student_baseline'])
 student={safe_key(r):load_packet(paired/f'{safe_key(r)}.pt') for r in rows}
 codecs=load_codecs(job);adapter=AlignedMetricAdapter(plans,codecs['student'],codecs['teacher'],require_dual=True)
 if len(rows)!=len(ids):raise ValueError('Missing frozen source IDs')
 for row,nat in zip(rows,native):
  plan=plans[safe_key(row)]
  if plan['source_token_ids']!=row['token_ids']:raise ValueError('Changed source token prefix')
  validate_native_score(baseline[safe_key(row)],len(nat['token_ids']),codecs['teacher'].support_size)
 return job,rows,native,plans,baseline,student,codecs,adapter


def make_specs(native,plans,baseline,case,vectors):
 if case=='no_sv_replay':return [False]*len(native)
 from vllm.model_hooks.steering.api import ApplySpec,VectorSpec,SteeringSpec
 from vllm.model_hooks.steering.payloads import DirectionVector
 config=read_json(ROOT/'exploration/config.json');spec=config['candidates'][case];bank=vectors[case]
 specs=[]
 for row in native:
  values=bank.get('global',bank.get(row['id']))
  if values is None:raise ValueError('Missing question-specific exploration direction')
  if case=='source_adaptive_protect':
   strengths=native_strengths(row,plans[row['key']],baseline[row['key']]['overlap'],case,spec)
   clauses=[];p=len(row['prompt_token_ids'])
   for alpha in (.2,.3):
    positions=(strengths==torch.tensor(alpha,dtype=strengths.dtype)).nonzero().flatten().tolist()
    if positions:clauses.append(VectorSpec(data=DirectionVector(values),layers=spec['layers'],
      algorithm='steer_opd_slerp',scale=alpha,normalize=False,
      apply=ApplySpec(prompt_positions=[p-1+t for t in positions]),name=f'{row["key"]}-{alpha}'))
   value=SteeringSpec(vectors=clauses) if clauses else False
  elif spec['position_mode']=='prediction':
   value=build_spec(row,[True]*len(row['token_ids']),values,alpha=spec['alpha'])
  else:
   # Explicit sv_training answer-input control, kept separate from the main prediction mask.
   p=len(row['prompt_token_ids']);n=len(row['token_ids'])
   value=SteeringSpec(vectors=[VectorSpec(data=DirectionVector(values),layers=spec['layers'],
     algorithm='steer_opd_slerp',scale=spec['alpha'],normalize=False,
     apply=ApplySpec(prompt_positions=list(range(p,p+n))),name=f'exploration-{row["key"]}')])
  specs.append(value)
 return specs


def records(rows,native,plans,scores,metrics,baseline,case,config):
 spec=config['candidates'].get(case);result={}
 for row,nat,score,(key,metric) in zip(rows,native,scores,metrics):
  if safe_key(row)!=key:raise ValueError('Metric order changed')
  valid=baseline[key]['alignment_mask']
  strengths=native_strengths(nat,plans[key],baseline[key]['overlap'],case,spec)
  native_mask=strengths>0
  mapped=torch.zeros(len(row['token_ids']),dtype=torch.bool)
  for u in plans[key]['units']:
   if not u['valid']:continue
   a,b=u['ranges']['student'];c,d=u['ranges']['teacher'];mapped[a:b]=bool(native_mask[c:d].any())
  condition={'id':case,'direction':'S-T','operator':'slerp','steer_role':'teacher','gate':'all',
   'granularity':'global' if case!='knn_shrink_global' else 'knn64_global_shrink',
   'alpha':spec['alpha'],'layers':spec['layers'],'position_mode':spec['position_mode']} if spec else None
  result[key]={**score,**metric,'mask_id':baseline[key]['mask_id'],'source_binding':baseline[key]['source_binding'],
   'condition':condition,'vector_id':config['directions_sha256'] if spec else None,
   'intervention_role':'teacher','metric_position_role':'student','steering_mask_semantics':'teacher_span_mapped_to_student',
   'requested_mask':mapped,'steering_mask':mapped,'native_requested_mask':native_mask,'native_steering_mask':native_mask,
   'native_prediction_positions':len(nat['token_ids']),'relative_perturbation_available':False,
   'strength_policy':'source_adaptive_frozen' if case=='source_adaptive_protect' else 'fixed',
   'native_effective_alpha':strengths,'effective_alpha':float(strengths.mean()),'base_alpha':spec['alpha'] if spec else 0.,
   'binding':digest({'case':case,'config':digest(config),'row_key':key,'original_mask_id':baseline[key]['mask_id']}),
   'fixed_student_identity':baseline[key]['fixed_student_identity'],'parameter_updates':0,
   'delta_overlap':metric['overlap']-baseline[key]['overlap'],'delta_P_overlap':metric['P_overlap']-baseline[key]['P_overlap']}
 return result


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['pilot','select','confirm'],required=True)
 parser.add_argument('--cases',required=True);args=parser.parse_args()
 configure_seed(42);torch.set_num_threads(8)
 config=read_json(ROOT/'exploration/config.json');cases=args.cases.split(',')
 if file_sha256(input_path(config['directions_path']))!=config['directions_sha256']:raise ValueError('Directions changed')
 vectors=load_packet(input_path(config['directions_path']))
 job,rows,native,plans,baseline,student,codecs,adapter=load_data(config,args.phase)
 output=OUT/args.phase;summary={};safety={}
 with Progress(output/'logs','exploration_'+args.phase,len(rows)*len(cases),unit='score',show_bar=False) as progress, MetricExecutor(adapter,workers=8,progress=progress) as executor:
  engine=None
  for case in cases:
   folder=output/case
   binding=digest({'config':config,'phase':args.phase,'case':case,'source_snapshot':read_json(ROOT/'exploration/snapshot_manifest.json'),
    'source_rows':digest(rows)})
   store=StageCheckpoint(folder,binding,tensor_compression='gzip');ready=store.load_tensors([safe_key(r) for r in rows])
   if ready is None and args.phase=='pilot' and case not in ('no_sv_replay','reference_replay') and config.get('reuse_pilot_from'):
    old_folder=input_path(config['reuse_pilot_from'])/'pilot'/case
    old_cfg=read_json(input_path(config['reuse_pilot_config']))
    for field in ('pilot_select_ids','original_split','data_sha256','teacher','student','metric_version','returned_top_k'):
     if old_cfg[field]!=config[field]:raise ValueError('Changed pilot reuse science '+field)
    if old_cfg['candidates'][case]!=config['candidates'][case]:raise ValueError('Pilot candidate changed')
    if (old_folder/'complete.json').exists():
     marker=read_json(old_folder/'stage_input.json')
     ready=StageCheckpoint(old_folder,marker['binding']).load_tensors([safe_key(r) for r in rows])
     if ready is not None:
      atomic_json(folder/'reused_commit.json',{'source':str(old_folder.relative_to(ROOT)),
       'complete_sha256':file_sha256(old_folder/'complete.json'),
       'old_config_sha256':file_sha256(input_path(config['reuse_pilot_config'])),
       'guard':'new no-SV and reference replay checks pass before referencing prior independent candidates'})
   if ready is None:
    if engine is None:
     engine=create_engine(model_path(job['teacher']),codecs['teacher'].support_size,
      max_model_len=max(len(r['prompt_token_ids'])+len(r['token_ids']) for r in native)+1,
      tensor_parallel_size=2,gpu_memory_utilization=.75,batch_invariant=False,seed=42,progress=progress)
    with progress.phase_scope('teacher_scoring',condition=case):
     scores=fixed_prefix_scores(engine,native,make_specs(native,plans,baseline,case,vectors),include_sampled=True,progress=progress)
    for r,v in zip(native,scores):validate_native_score(v,len(r['token_ids']),codecs['teacher'].support_size)
    with progress.phase_scope('cpu_metrics',condition=case):
     metrics=executor.compute([(safe_key(r),student[safe_key(r)],v) for r,v in zip(rows,scores)])
     ready=records(rows,native,plans,scores,metrics,baseline,case,config)
    if case in ('no_sv_replay','reference_replay'):
     expected=baseline if case=='no_sv_replay' else {safe_key(r):load_packet(ST/'select/teacher/scheme_05'/f'{safe_key(r)}.pt') for r in rows}
     delta_mean=torch.tensor([float((ready[k]['overlap']-expected[k]['overlap'])[ready[k]['alignment_mask']].mean())*100 for k in ready]).mean().item()
     logp_mae=torch.cat([(ready[k]['sampled_logps']-expected[k]['sampled_logps']).abs() for k in ready]).mean().item()
     check={'mean_overlap_difference_pp':delta_mean,'sampled_logp_mae':logp_mae,
      'bounds':{'absolute_mean_overlap_difference_pp':.1,'sampled_logp_mae':.05}}
     atomic_json(folder/'replay_check.json',check)
     if abs(delta_mean)>.1 or logp_mae>.05:raise ValueError(f'Replay comparability failed: {case} {check}')
    with progress.phase_scope('save_complete_candidate',condition=case):store.save_tensors(ready)
    progress.mark_committed(progress.committed+len(rows))
   else:progress.resume(progress.done+len(rows))
   metrics=summarize_teacher(rows,ready,baseline)
   summary[case]={'metrics':metrics,'pilot_only':args.phase=='pilot','questions':len(rows),
    'condition':config['candidates'].get(case),'rank_metric':'overlap'}
   atomic_json(output/'summaries.json',summary)
   print('RESULT',args.phase,case,{m:metrics[m]['question_weighted_delta_pp'] for m in metrics},flush=True)
   if case not in ('no_sv_replay','negative_direction_control'):
    with progress.phase_scope('plots',condition=case):plot_dual_comparisons(rows,baseline,ready,folder/'comparison',requested_keys=[safe_key(r) for r in rows[:3]])
   del ready
  atomic_json(output/'complete.json',{'status':'complete','phase':args.phase,'cases':cases,'questions':len(rows),'parameter_updates':0})
if __name__=='__main__':main()
