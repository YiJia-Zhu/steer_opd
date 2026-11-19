"""Pre-update Top16 overlap, reusing student rollout and steered teacher scores."""
from concurrent.futures import ProcessPoolExecutor
import gzip
import multiprocessing
import os
from pathlib import Path
import time

import torch

from steer_opd.compass_alignment import aligned_metrics
from steer_opd.prefix_matching import METRIC_VERSION,RULE_VERSION,P_DEFINITION
from steer_opd.resumable_core import ROOT,atomic_json
from steer_opd.static_masks import validate_native_score
from steer_opd.retention import minimal,keep_model_history


def save_top16(path,records):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.tmp.{os.getpid()}')
    try:
        with gzip.open(temporary,'wb',compresslevel=1) as stream:torch.save(records,stream)
        os.replace(temporary,path)
    finally:temporary.unlink(missing_ok=True)


def load_top16(path):
    with gzip.open(path,'rb') as stream:return torch.load(stream,map_location='cpu',weights_only=True)


def student_records(batch):
    extras=batch.non_tensor_batch;responses=batch.batch['responses'];width=responses.shape[1]
    if 'response_topk_ids' not in extras or 'response_topk_log_probs' not in extras:
        raise ValueError('Current rollout has no Top16; never substitute another checkpoint or prefix')
    records=[]
    for i in range(len(responses)):
        n=int(batch.batch['attention_mask'][i,-width:].sum())
        ids=torch.as_tensor(extras['response_topk_ids'][i],dtype=torch.int32)
        logps=torch.as_tensor(extras['response_topk_log_probs'][i],dtype=torch.float32)
        if ids.shape!=(n,16) or logps.shape!=ids.shape:raise ValueError('Student Top16 response length changed')
        records.append({'ids':ids,'logps':logps,
            'sampled_logps':batch.batch['rollout_log_probs'][i,:n].detach().float().cpu(),
            'source':'vllm_rollout_raw_logprobs','probability_mode':'raw_logprobs'})
    return records


def row_statistics(row,alignment,student,teacher,codecs):
    validate_native_score(student,len(row['token_ids']),codecs['student'].support_size)
    result=aligned_metrics(alignment,student,teacher,codecs['student'],codecs['teacher'])
    valid=torch.isfinite(result['overlap']) & torch.isfinite(result['P_overlap'])
    # Conditional Top16 entropy at valid native student positions, distinct
    # from full-vocabulary actor entropy and span-start broadcast statistics.
    conditional=student['logps'].double().softmax(dim=-1)
    h16=torch.special.entr(conditional).sum(-1)
    return {'question_id':row['question_id'],'trajectory_id':row['id'],
        'invalid_utf8_bytes':alignment.get('invalid_utf8',{}).get('bytes',0),
        'total_tokens':len(row['token_ids']),'valid_tokens':int(valid.sum()),
        'student_H16_cond_sum':float(h16[valid].sum()),
        'overlap_sum':float(result['overlap'][valid].sum()),
        'P_overlap_sum':float(result['P_overlap'][valid].sum())}


def summarize_statistics(parts):
    groups={};valid=sum(p['valid_tokens'] for p in parts);total=sum(p['total_tokens'] for p in parts)
    if not valid:raise ValueError('No aligned positions for training overlap')
    for part in parts:
        g=groups.setdefault(part['question_id'],{'valid_tokens':0,'overlap_sum':0.,'P_overlap_sum':0.})
        for key in g:g[key]+=part[key]
    groups={qid:g for qid,g in groups.items() if g['valid_tokens']}
    metrics={'overlap/coverage':valid/max(1,total),'overlap/valid_tokens':valid,'overlap/questions':len(groups)}
    metrics['alignment/invalid_utf8_bytes']=sum(p.get('invalid_utf8_bytes',0) for p in parts)
    metrics['alignment/invalid_utf8_trajectories']=sum(p.get('invalid_utf8_bytes',0)>0 for p in parts)
    for name in ('overlap','P_overlap'):
        metrics[name+'/token_value']=sum(p[name+'_sum'] for p in parts)/valid
        metrics[name+'/question_value']=sum(g[name+'_sum']/g['valid_tokens'] for g in groups.values())/len(groups)
    metrics['rollout/H16_cond_native']=sum(p['student_H16_cond_sum'] for p in parts)/valid
    return metrics


_CODECS=None


def _initialize_worker(paths):
    global _CODECS
    os.environ['CUDA_VISIBLE_DEVICES']='';torch.set_num_threads(1)
    from transformers import AutoTokenizer
    from steer_opd.compass_alignment import ByteLevelCodec
    _CODECS={role:ByteLevelCodec(AutoTokenizer.from_pretrained(ROOT/path,local_files_only=True)) for role,path in paths.items()}


def _work(packet):
    return row_statistics(*packet,_CODECS)


class OverlapMonitor:
    def __init__(self,plan,codecs):
        self.plan=plan;self.codecs=codecs;self.pool=None

    def compute(self,batch,rows,replies,step):
        began=time.monotonic();student=student_records(batch)
        if len(rows)!=len(replies) or len(rows)!=len(student):raise ValueError('Overlap trajectory order changed')
        packets=[]
        for row,s,reply in zip(rows,student,replies):
            if reply['alignment']['source_token_ids']!=row['token_ids']:raise ValueError('Overlap prefix differs from OPD')
            if 'teacher_top16' not in reply:raise ValueError('Same-call teacher Top16 is unavailable')
            packets.append((row,reply['alignment'],s,reply['teacher_top16']))
        workers=min(len(packets),max(1,int(os.environ.get('STEER_OPD_METRIC_WORKERS','16'))))
        if workers==1:parts=[row_statistics(*p,self.codecs) for p in packets]
        else:
            if self.pool is None:
                paths={role:self.plan['tokenizer_paths'][role] for role in ('student','teacher')}
                self.pool=ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
                    initializer=_initialize_worker,initargs=(paths,))
            parts=list(self.pool.map(_work,packets,chunksize=2))
        metrics=summarize_statistics(parts);metrics['overlap/cpu_wall_seconds']=time.monotonic()-began
        folder=ROOT/self.plan['output']/f'overlap/step_{step:06d}'
        if not minimal(self.plan):save_top16(folder/'student_top16.pt.gz',student)
        target=(ROOT/self.plan['output']/'overlap/latest.json' if minimal(self.plan) and keep_model_history(self.plan)
                else folder/'summary.json')
        atomic_json(target,{'step':step,'student_policy_step':step-1,
            'measurement':'same_rollout_pre_update','teacher':'actual_S-T_global_steered_teacher',
            'metric_version':METRIC_VERSION,'matching_rule_version':RULE_VERSION,
            'P_overlap_definition':P_DEFINITION,'normalization':'original_probabilities_no_top16_renormalization',
            'entropy_definition':'Top16 conditional entropy in nats, valid native student positions; not full-vocabulary entropy',
            'extra_generation_calls':0,'extra_model_forward_calls':0,'plan_binding':self.plan['binding'],
            'metrics':metrics,**({} if minimal(self.plan) else {'trajectories':parts})})
        return metrics

    def close(self):
        if self.pool is not None:self.pool.shutdown(wait=True)
