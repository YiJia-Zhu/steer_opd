#!/usr/bin/env python3
"""Freeze 4,096 SFT questions and the held teacher request for the 3,072 added questions."""
import argparse
from collections import Counter
import fcntl
import hashlib
import io
import os
from pathlib import Path
import subprocess
import sys
from urllib.request import urlopen
import uuid

from launch_ablation import run_validation
from prepare_ablation_data import local_candidates, question_hash, select_expansion
from steer_opd.compass_campaign import frozen_json, generation_job
from steer_opd.compass_job import load_codecs
from steer_opd.direction_bank import file_sha256
from steer_opd.execution_policy import execution_policy, require_execution_allowed
from steer_opd.experiment_runtime import child_environment
from steer_opd.resumable_core import ROOT, digest, read_json
from steer_opd.sft_baseline import DEFAULT_REFERENCE_ANSWERS, DEFAULT_TEACHER_DIR, EXPANDED_RUN_NAME, teacher_answers
from steer_opd.stage_io import component_target, configure_layout, project
from steer_opd.training.benchmarks import benchmark_rows
from steer_opd.training.verl_plan import gpu_ids


QUESTION_COUNT = 4096
PARAM_NAME = 'n4096_sft_seed42_v1'
REFERENCE_TABLE_BYTES = 73445749


def reference_table(source):
    import pyarrow.parquet as pq

    cache = ROOT / '.runtime/sft_reference_tables' / f"{source['sha256']}.parquet"
    if cache.is_file():
        data = cache.read_bytes()
    else:
        url = (f"https://huggingface.co/datasets/{source['dataset']}"
               f"/resolve/{source['revision']}/{source['path']}?download=true")
        with urlopen(url, timeout=60) as response:
            data = response.read(REFERENCE_TABLE_BYTES + 1)
    if len(data) != REFERENCE_TABLE_BYTES or hashlib.sha256(data).hexdigest() != source['sha256']:
        raise ValueError(f'Published SFT reference table size or SHA256 mismatch ({len(data)} bytes)')
    return pq.ParquetFile(io.BytesIO(data))


def published_answers(parquet, column, questions):
    found = {}
    conflicts = set()
    for batch in parquet.iter_batches(columns=['problem', column, 'problem_source']):
        for raw, answer, origin in zip(*(column.to_pylist() for column in batch.columns), strict=True):
            question = raw.strip()
            if question in questions and isinstance(answer, str) and answer.strip():
                value = (answer.strip(), origin)
                if question in found and found[question][0] != value[0]:
                    conflicts.add(question)
                found.setdefault(question, value)
    for question in conflicts:
        found.pop(question)
    return found, len(conflicts)


def prepare(cfg):
    base_file = project(cfg['training_input_data_dir']) / 'questions_source.json'
    base = read_json(base_file)
    base_sha = file_sha256(base_file)
    if len(base) != 1024 or len({q['id'] for q in base}) != len(base):
        raise ValueError('Expected the frozen 1,024 OPD questions as the SFT expansion prefix')
    old_refs = read_json(ROOT / DEFAULT_REFERENCE_ANSWERS)
    if old_refs['frozen_questions_sha256'] != base_sha:
        raise ValueError('The original SFT reference table belongs to a different question pool')
    old_teacher = ROOT / DEFAULT_TEACHER_DIR
    original_job = read_json(old_teacher / 'job.json')['job']
    teacher_answers(old_teacher, {q['id']: q['question'] for q in base}, cfg['teacher'],
                    original_job['model_content_id'])

    cfg = configure_layout({**cfg, 'param_name': PARAM_NAME})
    data = project(cfg['data_dir'])
    step1 = project(cfg['step_dirs']['1'])
    held_job = step1 / 'metadata/jobs/teacher_held.json'
    teacher_dir = project(component_target(cfg, 'teacher'))
    manifest_path = data / 'sampling_manifest.json'
    data.mkdir(parents=True, exist_ok=True)
    with (data / '.prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            if (manifest['base_source_sha256'] != base_sha or
                    any(file_sha256(project(path)) != sha for path, sha in manifest['files'].items())):
                raise ValueError('Frozen SFT expansion inputs changed')
            return manifest

        table = reference_table(old_refs['source'])
        print('Reading local OT3 math candidates and published answer references', flush=True)
        candidates, provenance = local_candidates(ROOT / cfg['local_dataset'], QUESTION_COUNT - len(base),
                                                  cfg['seed'], excluded_ids={q['id'] for q in base})
        references, conflicts = published_answers(table, old_refs['source']['column'],
                                                 {q['question'] for q in candidates})
        held_out, _ = benchmark_rows(base)
        seen = {' '.join(q['question'].split()) for q in base}
        seen.update(' '.join(row['extra_info']['question'].split()) for row in held_out)
        eligible = []
        for row in candidates:
            question = row['question']
            key = ' '.join(question.split())
            if question in references and key not in seen:
                eligible.append(row)
                seen.add(key)

        codecs = load_codecs(cfg)
        selected, rejected = select_expansion(base, eligible, QUESTION_COUNT, codecs,
                                              cfg['max_prompt_tokens'], cfg['enable_thinking'])
        added = selected[len(base):]
        questions = base + added
        benchmark_rows(questions)
        job = generation_job(cfg, codecs['teacher'], 'teacher', added)
        sampling_fields = ('model', 'tokenizer_paths', 'max_tokens', 'temperature', 'top_p',
                           'presence_penalty', 'seed', 'head_dtype', 'tensor_parallel_size')
        if any(job[key] != original_job[key] for key in sampling_fields):
            raise ValueError('Expanded teacher generation settings differ from the original SFT teacher job')
        job['bind_checkpoint_content'] = True
        job['execution_hold'] = cfg['execution_hold']
        job['model_content_id'] = original_job['model_content_id']

        questions_path = data / 'questions_source.json'
        refs_path = data / 'reference_answers.json'
        frozen_json(questions_path, questions)
        answers = {question_hash(q['question']): old_refs['answers'][question_hash(q['question'])] for q in base}
        answers.update({question_hash(q['question']): references[q['question']][0] for q in added})
        frozen_json(refs_path, {
            'schema': old_refs['schema'], 'source': old_refs['source'], 'answers': answers,
            'frozen_questions_sha256': file_sha256(questions_path),
            'added_problem_sources': dict(Counter(references[q['question']][1] for q in added)),
        })
        frozen_json(held_job, job)
        manifest = {
            'schema': 'sft_question_expansion_v1', 'questions': QUESTION_COUNT,
            'preserved_base_questions': len(base), 'added_questions': len(added),
            'base_source': str(base_file.relative_to(ROOT)), 'base_source_sha256': base_sha,
            'eligible_labeled_candidates': len(eligible), 'conflicting_reference_questions': conflicts,
            'rejected_candidates': rejected,
            'dataset_revision': cfg['dataset_revision'], 'sampled_source': provenance,
            'reference_answers': str(refs_path.relative_to(ROOT)),
            'teacher_sources': {'base': DEFAULT_TEACHER_DIR, 'added': str(teacher_dir.relative_to(ROOT))},
            'teacher_job': str(held_job.relative_to(ROOT)), 'teacher_job_hash': digest(job),
            'files': {str(path.relative_to(ROOT)): file_sha256(path) for path in (questions_path, refs_path, held_job)},
        }
        frozen_json(manifest_path, manifest)
        return manifest


def run_expansion(manifest, cfg, config_path, physical_gpus):
    held = project(manifest['teacher_job'])
    teacher_dir = project(manifest['teacher_sources']['added'])
    data_dir = project(manifest['reference_answers']).parent
    engine = str(ROOT / cfg['engine_python'])
    curand = ROOT / '.runtime/cuda-13.0-curand-headers'
    if (curand / 'curand.h').is_file():
        os.environ['CPATH'] = os.pathsep.join(filter(None, (str(curand), os.environ.get('CPATH'))))
    env = os.environ.copy()
    with (held.parent / '.sft_launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (teacher_dir / 'complete.json').is_file():
            validation = configure_layout({**cfg, 'param_name': PARAM_NAME})
            validation.update(execution_hold=None, execution_blockers=[], gpu_wait_seconds=3600,
                              optimizer_updates=0, stop_after='source_overlap',
                              gpu_acceptance_receipt=str(component_target(validation, 'acceptance') / 'passed.json'))
            approved = run_validation(validation, physical_gpus, held.with_name('teacher_acceptance_config.json'))
            job = {**read_json(held), **execution_policy(approved)}
            require_execution_allowed(job)
            executable = held.with_name('teacher_run.json')
            frozen_json(executable, job)
            attempt = held.parent.parent / 'attempts' / uuid.uuid4().hex[:12]
            subprocess.run([
                sys.executable, 'scripts/run_on_idle_gpus.py', '--count', '2', '--gpu-ids', physical_gpus,
                '--wait-seconds', str(validation['gpu_wait_seconds']), '--run-dir', str(attempt), '--',
                engine, 'scripts/generate_static_sources.py', '--job', str(executable),
            ], cwd=ROOT, env=child_environment(engine), check=True)
        for key in ('SFT_TOTAL_STEPS', 'SFT_TEACHER_DIR', 'SFT_REFERENCE_ANSWERS'):
            env.pop(key, None)
        env.update(METHODS='sft', SFT_DATA_DIR=str(data_dir), SFT_RUN_NAME=EXPANDED_RUN_NAME,
                   GPU_IDS=physical_gpus, PREPARED_CONFIG=str(config_path), TRAIN_BATCH_SIZE='32')
        subprocess.run(['bash', 'scripts/baselines.sh', '--run'], cwd=ROOT, env=env, check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/ablation.json')
    parser.add_argument('--run', action='store_true', help='Generate added teacher responses and then launch SFT')
    parser.add_argument('--gpu-ids', default='6,7')
    args = parser.parse_args()
    if args.run and len(gpu_ids(args.gpu_ids)) != 2:
        parser.error('SFT expansion uses two distinct physical GPUs')
    config_path = project(args.config)
    experiment = read_json(config_path)
    result = prepare(experiment)
    print({key: result[key] for key in ('questions', 'preserved_base_questions', 'added_questions',
                                      'teacher_sources', 'teacher_job')}, flush=True)
    if args.run:
        run_expansion(result, experiment, config_path, args.gpu_ids)
