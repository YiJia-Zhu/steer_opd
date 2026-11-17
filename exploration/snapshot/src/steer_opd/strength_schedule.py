"""Frozen per-question strength assignments for zero-update annealing diagnostics."""
import math

from steer_opd.operators import validate_operator
from steer_opd.resumable_core import digest

VERSION = 'linear_80_tail20_v1'
LINEAR = 'linear_80_tail20'
POLICIES = ('fixed', LINEAR)


def schedule_value(base_alpha, index, *, total=100, cutoff=80, policy=LINEAR):
    validate_operator('slerp', base_alpha)
    if (type(total) is not int or type(cutoff) is not int or not 0 < cutoff < total
            or type(index) is not int or not 0 <= index < total or policy not in POLICIES):
        raise ValueError('Invalid frozen strength schedule index/cutoff/policy')
    if policy == LINEAR:
        validate_operator('slerp', 2*base_alpha)
    return float(base_alpha if policy == 'fixed' else 2*base_alpha*max(1-index/cutoff, 0.))


def freeze_schedule(questions, *, cutoff_fraction=.8, seed=42):
    ids=[q['id'] for q in questions]
    if (seed != 42 or len(ids)<2 or len(set(ids))!=len(ids)
            or not math.isfinite(cutoff_fraction) or not 0<cutoff_fraction<1):
        raise ValueError('Need unique questions and a valid frozen strength schedule')
    # Independent hash order avoids using the same seed42 shuffle as select/confirm.
    # It never reads either split or model/metric outcomes.
    order=sorted(ids,key=lambda qid:(digest({'seed':seed,'domain':VERSION,'question_id':qid}),qid))
    cutoff=math.floor(len(order)*cutoff_fraction)
    if not 0<cutoff<len(order):raise ValueError('Too few questions for a nonempty annealing/tail partition')
    value={'version':VERSION,'scope':'frozen_question_assignment_zero_update_diagnostic',
        'order_rule':'independent_seed42_question_hash_before_filtering_and_split',
        'total':len(order),'cutoff':cutoff,'cutoff_fraction':cutoff_fraction,'seed':seed,
        'question_pool_sha256':digest(sorted(questions,key=lambda q:q['id'])),
        'question_indices':{qid:i for i,qid in enumerate(order)}}
    value['schedule_id']=digest(value)
    return value


def validate_schedule(value):
    if (value.get('version')!=VERSION or value.get('schedule_id')!=digest({k:v for k,v in value.items() if k!='schedule_id'})
            or type(value.get('total')) is not int or type(value.get('cutoff')) is not int
            or not 0<value['cutoff']<value['total']
            or any(type(i) is not int for i in value['question_indices'].values())
            or sorted(value['question_indices'].values())!=list(range(value['total']))):
        raise ValueError('Frozen strength schedule content/identity changed')
    return value


def row_strength(condition, row, schedule=None):
    if condition is None:
        return {'strength_policy':'none','base_alpha':0.,'effective_alpha':0.,'schedule_index':None,'schedule_id':None}
    policy=condition.get('strength_policy','fixed')
    if policy not in POLICIES:raise ValueError('Unknown strength policy')
    base=condition.get('alpha',.1)
    validate_operator(condition.get('operator','slerp'),base)
    if policy=='fixed':
        result={'strength_policy':policy,'base_alpha':float(base),'effective_alpha':float(base),
                'schedule_index':None,'schedule_id':None}
        if schedule is not None:
            if row['id'] not in schedule['question_indices']:raise ValueError('Question absent from frozen strength schedule')
            result.update(schedule_index=schedule['question_indices'][row['id']],schedule_id=schedule['schedule_id'],
                schedule_total=schedule['total'],schedule_cutoff=schedule['cutoff'],schedule_scope=schedule['scope'])
        return result
    if condition.get('operator','slerp')!='slerp':raise ValueError('Annealing screen currently uses Slerp alpha')
    if schedule is None:raise ValueError('Annealed candidate requires a frozen full-pool schedule')
    if condition.get('schedule_id')!=schedule['schedule_id']:
        raise ValueError('Candidate strength schedule identity mismatch')
    try:index=schedule['question_indices'][row['id']]
    except KeyError:raise ValueError('Question absent from frozen strength schedule') from None
    alpha=schedule_value(base,index,total=schedule['total'],cutoff=schedule['cutoff'],policy=policy)
    return {'strength_policy':policy,'base_alpha':float(base),'effective_alpha':alpha,'schedule_index':index,
            'schedule_id':schedule['schedule_id'],'schedule_total':schedule['total'],'schedule_cutoff':schedule['cutoff'],
            'schedule_scope':schedule['scope']}


def confirm_with_counterparts(ranked, conditions, top_n):
    by_id={c['id']:c for c in conditions};chosen=list(ranked[:top_n])
    for key in list(chosen):
        peer=by_id[key].get('paired_condition_id')
        if peer in by_id and peer not in chosen:chosen.append(peer)
    return chosen
