"""Frozen gold-labelled benchmark monitoring, separate from OPD rewards."""
import json

from steer_opd.resumable_core import ROOT
from steer_opd.direction_bank import file_sha256
from steer_opd.math_data import TASK_SUFFIX


def benchmark_rows(training_questions):
    seen={' '.join(row['question'].split()) for row in training_questions}
    records=[];sources={};counts={}
    for name,relative,key,expected in [
        ('MATH-500','MATH-500/test.jsonl','problem',500),
        ('AIME2025','AIME2025/aime2025-I.jsonl','question',15),
        ('AIME2025','AIME2025/aime2025-II.jsonl','question',15),
    ]:
        path=ROOT/'assets/data'/relative
        rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(rows)!=expected:raise ValueError(f'Incomplete benchmark: {relative}')
        sources[str(path.relative_to(ROOT))]=file_sha256(path)
        for i,row in enumerate(rows):
            question=row[key];answer=str(row['answer']).strip()
            if not question or not answer:raise ValueError('Missing benchmark question or gold answer')
            if ' '.join(question.split()) in seen:
                raise ValueError(f'Benchmark duplicates an SV extraction question: {relative}:{i}')
            qid=f'{relative}:{i}'
            records.append({'data_source':'steer_eval_'+name,
                'prompt':[{'role':'user','content':question+TASK_SUFFIX}],
                'ability':'math','reward_model':{'style':'rule','ground_truth':answer},
                'extra_info':{'question_id':qid,'question':question,'split':'benchmark','benchmark':name}})
            counts[name]=counts.get(name,0)+1
    if len({r['extra_info']['question'] for r in records})!=530:raise ValueError('Duplicate benchmark questions')
    return records,{'counts':counts,'sources':sources,'samples_per_question':1,
                    'metric':'single_sample_accuracy','grader':'pinned_verl.math_verify',
                    'exact_question_overlap_with_sv_questions':0,'sv_question_count':len(training_questions),
                    'excluded_from_training':True,'student_steering':False}
