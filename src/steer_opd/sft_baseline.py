"""Teacher-trajectory SFT using the pinned VERL trainer and the frozen math pool."""
import argparse
import fcntl
from itertools import islice
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from tempfile import NamedTemporaryFile, TemporaryDirectory
import uuid

from steer_opd.checkpoint_identity import checkpoint_content_id
from steer_opd.direction_bank import file_sha256
from steer_opd.evaluation_text import FORMAT, response_path, responses_complete, save_responses
from steer_opd.experiment_runtime import StageCheckpoint, child_environment, configure_seed
from steer_opd.generation_policy import DEFAULT_MAX_GENERATION_LENGTH, DEFAULT_PRESENCE_PENALTY, presence_penalty
from steer_opd.math_data import prompt_ids
from steer_opd.resumable_core import ROOT, atomic_json, digest, read_json
from steer_opd.retention import ALL_MODELS_LATEST_STATE, MINIMAL
from steer_opd.stage_io import project
from steer_opd.training.benchmarks import benchmark_rows
from steer_opd.training.paths import training_output_dir
from steer_opd.training.retention import (
    SFT_LAYOUT, commit_checkpoint, prune_checkpoints, recover_checkpoint, validate_model_checkpoint,
)
from steer_opd.training.storage import require_training_space
from steer_opd.training.telemetry import ResourceMonitor, phase
from steer_opd.training.verl_entry import activate_verl, admitted_pool, validate_plan
from steer_opd.training.verl_plan import PAIR, VERL_REVISION, VERL_SOURCE, gpu_ids


DEFAULT_TEACHER_DIR = (
    'outputs/step_1_rollout/ablation/' + PAIR
    + 'n1024_slerp0.1_S26_T29_seed42_trainall_Tgen16384_Sgen16384_Ttemp1_Tp1_pp2_v1_st_global_all_full_v3/teacher'
)
MAX_LENGTH = 32768
ENTRY = 'steer_opd.sft_baseline'


def teacher_answers(directory, questions, teacher, content_id):
    """Read only normally terminated, non-empty answers from a committed teacher call."""
    identity = read_json(directory / 'stage_input.json')
    complete = StageCheckpoint(directory, identity['binding']).load()
    if complete is None or complete['requests'] != len(questions) or 'job.json' not in complete['files']:
        raise ValueError('SFT requires a committed teacher response for the complete frozen question pool')
    record = read_json(directory / 'job.json')
    job = record['job']
    if (record['hash'] != digest(job) or job['role'] != 'teacher'
            or job['model_content_id'] != content_id or job.get('save_rollout_top_k')):
        raise ValueError('SFT source is not the frozen teacher generation job')
    requests = {row['id']: row for row in job['requests']}
    if len(job['requests']) != len(questions) or requests.keys() != questions.keys():
        raise ValueError('Teacher rollout question IDs differ from the frozen training pool')
    answers = {}
    for qid, question in questions.items():
        request = requests[qid]
        name = f"{request['key']}.json"
        if Path(name).name != name or name not in complete['files'] or request['question'] != question:
            raise ValueError(f'Teacher response is missing or belongs to a different question: {qid}')
        row = read_json(directory / name)
        if (row['id'] != qid or row['question'] != question or row['job_hash'] != record['hash']
                or row['prompt_token_ids'] != request['prompt_token_ids'] or row['steering'] is not False
                or row['model'] != teacher or row['model_content_id'] != content_id):
            raise ValueError(f'Teacher response identity changed: {qid}')
        if row['finish_reason'] == 'stop' and row['text'].strip():
            answers[qid] = row['text']
    if not answers:
        raise ValueError('SFT has no normally terminated, non-empty teacher responses')
    return answers, record['hash']


def encode_example(tokenizer, qid, question, text):
    # Append the original response to the rollout prompt. An assistant chat template
    # can parse away text before a literal </think> in the teacher's answer.
    prefix = prompt_ids(tokenizer, question, enable_thinking=False)
    answer = tokenizer.encode(text, add_special_tokens=False)
    if not prefix or not answer or tokenizer.eos_token_id is None:
        raise ValueError(f'Invalid SFT prompt, response or EOS: {qid}')
    ids = prefix + answer + [tokenizer.eos_token_id]
    if len(ids) > MAX_LENGTH:
        raise ValueError(f'SFT example {qid} exceeds the {MAX_LENGTH}-token context')
    return {'question_id': qid, 'input_ids': ids, 'prompt_length': len(prefix)}


def one_epoch_batches(examples, batch_size, ranks, requested_steps=None):
    per_rank = math.ceil(examples / ranks)
    limit = batch_size // ranks
    minimum = math.ceil(per_rank / limit)
    steps = minimum if requested_steps is None else requested_steps
    if not minimum <= steps <= per_rank:
        raise ValueError(f'One SFT epoch at batch size {batch_size} supports {minimum} to {per_rank} updates')
    # Keep full batches where possible and divide the remaining examples evenly.
    full = min(steps, (per_rank - steps) // max(1, limit - 1))
    tail_steps = steps - full
    if not tail_steps:
        return [batch_size] * full
    size, remainder = divmod(per_rank - full * limit, tail_steps)
    return [batch_size] * full + [ranks * (size + 1)] * remainder + [ranks * size] * (tail_steps - remainder)


def build_plan(args):
    from transformers import AutoTokenizer

    pool = gpu_ids(args.gpu_ids)
    if len(pool) != 2 or args.batch_size < 2 or args.batch_size % 2:
        raise ValueError('SFT uses two physical GPUs and a global batch size divisible by two')
    if args.save_freq < 1 or args.wait_seconds < 0:
        raise ValueError('SFT requires a positive save interval and a nonnegative GPU wait time')
    experiment = read_json(project(args.config))
    if args.train_output_root:
        experiment['train_output_root'] = args.train_output_root
    source = project(experiment.get('training_input_data_dir') or experiment['input_data_dir'])
    questions_path = source / 'questions_source.json'
    original = read_json(questions_path)
    questions = {row['id']: row['question'] for row in original}
    if len(original) != 1024 or len(questions) != 1024:
        raise ValueError('Use the complete frozen 1024-question training pool')
    benchmarks, evaluation = benchmark_rows(original)
    folder = project(args.teacher_dir)
    teacher = experiment['teacher']
    teacher_id = checkpoint_content_id(ROOT / teacher)
    answers, teacher_job = teacher_answers(folder, questions, teacher, teacher_id)
    student = experiment['student']
    tokenizer = AutoTokenizer.from_pretrained(ROOT / experiment['tokenizer_paths']['student'], local_files_only=True)
    rows = [encode_example(tokenizer, qid, questions[qid], text) for qid, text in answers.items()]
    batches = one_epoch_batches(len(rows), args.batch_size, len(pool), args.total_steps)
    steps = len(batches)
    paths = [questions_path, folder / 'stage_input.json', folder / 'complete.json']
    inputs = {str(path.relative_to(ROOT)): file_sha256(path) for path in paths} | evaluation['sources']
    files = (
        Path(__file__).resolve(), ROOT / 'src/steer_opd/evaluation_text.py', ROOT / 'src/steer_opd/math_data.py',
        ROOT / 'src/steer_opd/training/benchmarks.py', ROOT / 'src/steer_opd/training/retention.py',
        ROOT / 'src/steer_opd/training/verl_reward.py', ROOT / 'src/steer_opd/generation_policy.py',
    )
    plan = {
        'schema': 'steer_sft_baseline_v1', 'baseline_method': 'sft', 'reference': 'arXiv:2609.10154v1 Appendix A.2',
        'output': str(training_output_dir(experiment, args.run_name, PAIR)),
        'student': student, 'student_checkpoint_content_id': checkpoint_content_id(ROOT / student),
        'student_tokenizer': experiment['tokenizer_paths']['student'],
        'teacher_source': {'directory': str(folder.relative_to(ROOT)), 'job_hash': teacher_job,
                           'model_content_id': teacher_id, 'filter': 'normally_terminated_nonempty'},
        'training_questions': len(questions), 'sft_examples': len(rows),
        'max_observed_length': max(len(row['input_ids']) for row in rows),
        'excluded_examples': len(questions) - len(rows),
        'sampler_padding_examples': (-len(rows)) % len(pool),
        'input_sha256': inputs,
        'implementation_sha256': {str(path.relative_to(ROOT)): file_sha256(path) for path in files},
        'verl_revision': VERL_REVISION, 'seed': 42, 'teacher_steering': False,
        'checkpoint_layout': SFT_LAYOUT, 'checkpoint_retention': ALL_MODELS_LATEST_STATE, 'retention_policy': MINIMAL,
        'evaluation': {**evaluation, 'steps': [*range(0, steps, args.save_freq), steps]},
        'evaluation_response_format': FORMAT,
        'settings': {'gpu_pool': list(pool), 'train_batch_size': args.batch_size, 'total_steps': steps,
                     'batch_sizes': batches,
                     'save_freq': args.save_freq, 'gpu_wait_seconds': args.wait_seconds, 'run_name': args.run_name,
                     'epochs': 1, 'learning_rate': 2e-5, 'weight_decay': 1e-6, 'warmup_ratio': .1,
                     'scheduler': 'cosine', 'max_length': MAX_LENGTH,
                     'presence_penalty': presence_penalty({'presence_penalty': args.presence_penalty})},
    }
    plan['binding'] = digest(plan)
    return plan, rows, benchmarks


def sft_config(plan):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    s = plan['settings']
    run = Path(plan['output'])
    with initialize_config_dir(config_dir=str(ROOT / VERL_SOURCE / 'verl/trainer/config'), version_base=None):
        config = compose(config_name='sft_trainer_engine')
    config.model.path = str(ROOT / plan['student'])
    config.model.tokenizer_path = str(ROOT / plan['student_tokenizer'])
    config.data.train_files = str(run / 'data/train.parquet')
    config.data.train_batch_size = s['train_batch_size']
    config.data.micro_batch_size_per_gpu = 1
    config.data.use_dynamic_bsz = False
    config.data.max_length = s['max_length']
    config.data.num_workers = 0
    config.optim.lr = s['learning_rate']
    config.optim.weight_decay = s['weight_decay']
    config.optim.lr_warmup_steps_ratio = s['warmup_ratio']
    config.optim.lr_scheduler_type = s['scheduler']
    config.trainer.default_local_dir = str(run / 'checkpoints')
    config.trainer.project_name = 'steer_opd_sft'
    config.trainer.experiment_name = s['run_name']
    config.trainer.seed = plan['seed']
    config.trainer.total_epochs = s['epochs']
    config.trainer.total_training_steps = s['total_steps']
    config.trainer.save_freq = s['save_freq']
    config.trainer.n_gpus_per_node = len(s['gpu_pool'])
    config.trainer.logger = ['console', 'tensorboard']
    OmegaConf.resolve(config)
    return config


def write_parquet(path, rows):
    import pyarrow as pa
    import pyarrow.parquet as pq

    with NamedTemporaryFile(dir=path.parent, prefix=f'.{path.name}.', delete=False) as target:
        temporary = Path(target.name)
    try:
        pq.write_table(pa.Table.from_pylist(rows), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_prepared(plan):
    validate_plan(plan)
    run = Path(plan['output'])
    manifest = read_json(run / 'metadata/data_manifest.json')
    files = manifest['files']
    if (manifest['plan_binding'] != plan['binding'] or set(files) != {'train.parquet', 'benchmarks.parquet'}
            or any(file_sha256(run / 'data' / name) != sha for name, sha in files.items())):
        raise ValueError('Frozen SFT parquet changed')
    return run


def prepare(plan, rows, benchmarks, config):
    from omegaconf import OmegaConf

    run = Path(plan['output'])
    frozen = run / 'metadata/plan.json'
    if frozen.exists():
        if read_json(frozen) != plan:
            raise ValueError('Use a new SFT run name when changing the frozen plan')
        validate_prepared(plan)
        return frozen
    (run / 'data').mkdir(parents=True, exist_ok=True)
    files = {}
    for name, values in [('train.parquet', rows), ('benchmarks.parquet', benchmarks)]:
        path = run / 'data' / name
        write_parquet(path, values)
        files[name] = file_sha256(path)
    atomic_json(run / 'metadata/verl_config.json', OmegaConf.to_container(config, resolve=True))
    atomic_json(run / 'metadata/data_manifest.json', {'plan_binding': plan['binding'], 'files': files})
    atomic_json(frozen, plan)
    return frozen


class TokenizedAnswers:
    def __init__(self, path):
        import pyarrow.parquet as pq

        self.rows = pq.read_table(path, columns=['input_ids', 'prompt_length']).to_pylist()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import torch

        row = self.rows[index]
        positions = torch.arange(len(row['input_ids']))
        return {'input_ids': torch.tensor(row['input_ids']), 'position_ids': positions,
                'loss_mask': (positions >= row['prompt_length']).long()}


def train_worker(plan):
    import torch
    from torch.utils.data import DistributedSampler
    from torchdata.stateful_dataloader import StatefulDataLoader
    from verl.trainer.sft_trainer import SFTTrainer
    from verl.utils.dataset.dataset_utils import SFTTensorCollator
    from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group

    class Trainer(SFTTrainer):
        def _build_dataset(self):
            self.train_dataset = TokenizedAnswers(self.config.data.train_files)
            self.val_dataset = None

        def _build_dataloader(self):
            world = self.engine.get_data_parallel_size()
            settings = plan['settings']
            self.global_batch_size = self.config.data.train_batch_size
            self.train_sampler = DistributedSampler(self.train_dataset, num_replicas=world,
                rank=self.engine.get_data_parallel_rank(), seed=plan['seed'], shuffle=True, drop_last=False)
            indices = iter(self.train_sampler)
            batches = [list(islice(indices, size // world)) for size in settings['batch_sizes']]
            if sum(map(len, batches)) != len(self.train_sampler):
                raise ValueError('SFT batches do not cover the frozen sampler')
            self.train_dataloader = StatefulDataLoader(self.train_dataset,
                batch_sampler=batches, collate_fn=SFTTensorCollator(self.config.data.pad_mode))
            self.val_dataloader = None

        def _build_ckpt_handler(self):
            super()._build_ckpt_handler()
            save = self.ckpt_handler.save_checkpoint
            root = Path(self.config.trainer.default_local_dir)

            def save_checkpoint(step):
                if self.rank == 0:
                    require_training_space(root, 48 * 1024**3)
                save(step)
                if self.rank == 0:
                    with phase(root.parent / 'logs', 'sft_checkpoint_commit', step=step):
                        receipt = commit_checkpoint(root, step, plan)
                        prune_checkpoints(root, step, plan, committed=receipt)
                torch.distributed.barrier()

            self.ckpt_handler.save_checkpoint = save_checkpoint

    initialize_global_process_group()
    try:
        configure_seed(plan['seed'])
        Trainer(sft_config(plan)).fit()
    finally:
        destroy_global_process_group()


def evaluate_worker(plan, step, model):
    import pyarrow.parquet as pq
    from steer_opd.training.verl_reward import compute_score

    run = Path(plan['output'])
    rows = pq.read_table(run / 'data/benchmarks.parquet').to_pylist()
    question_ids = [row['extra_info']['question_id'] for row in rows]
    if responses_complete(run, step):
        with response_path(run, step).open(encoding='utf-8') as stream:
            texts = {item['question_id']: item['text'] for item in map(json.loads, stream)}
    else:
        from steer_opd.experiment_runtime import engine_startup_guard
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        configure_seed(plan['seed'])
        tokenizer = AutoTokenizer.from_pretrained(ROOT / plan['student_tokenizer'], local_files_only=True)
        prompts = [prompt_ids(tokenizer, row['extra_info']['question'], enable_thinking=False) for row in rows]
        with engine_startup_guard():
            llm = LLM(model=str(model), tokenizer=str(ROOT / plan['student_tokenizer']), dtype='bfloat16',
                      tensor_parallel_size=1, max_model_len=max(map(len, prompts)) + DEFAULT_MAX_GENERATION_LENGTH,
                      generation_config='vllm', hf_overrides={'head_dtype': 'float32'}, seed=plan['seed'])
        params = SamplingParams(n=1, seed=plan['seed'], temperature=1., top_p=1., top_k=-1,
            max_tokens=DEFAULT_MAX_GENERATION_LENGTH, presence_penalty=plan['settings']['presence_penalty'])
        outputs = llm.generate([{'prompt_token_ids': ids} for ids in prompts], params)
        if any(list(output.prompt_token_ids) != prompt or len(output.outputs) != 1
               for prompt, output in zip(prompts, outputs, strict=True)):
            raise ValueError('Incomplete or mismatched SFT benchmark generation')
        texts = dict(zip(question_ids, (output.outputs[0].text for output in outputs), strict=True))
        save_responses(run, step, question_ids, [texts[qid] for qid in question_ids])
    correct = dict.fromkeys(plan['evaluation']['counts'], 0)
    for row, qid in zip(rows, question_ids, strict=True):
        correct[row['extra_info']['benchmark']] += compute_score(
            row['data_source'], texts[qid], row['reward_model']['ground_truth'])['acc']
    metrics = {f'eval/{name}/acc': count / plan['evaluation']['counts'][name] for name, count in correct.items()}
    atomic_json(run / f'evaluation/metrics/step_{step}.json', metrics)


def evaluation_complete(plan, step):
    run = Path(plan['output'])
    path = run / f'evaluation/metrics/step_{step}.json'
    return (path.is_file() and set(read_json(path)) == {f'eval/{key}/acc' for key in plan['evaluation']['counts']}
            and responses_complete(run, step))


def execute(plan, frozen, admission):
    run = Path(plan['output'])
    logs = run / 'logs'
    checkpoints = run / 'checkpoints'
    steps = plan['evaluation']['steps']
    student = ROOT / plan['student']
    if checkpoint_content_id(student) != plan['student_checkpoint_content_id']:
        raise ValueError('Frozen initial student changed')
    command = [sys.executable, '-m', ENTRY]
    train_env = {**os.environ, 'VLLM_BATCH_INVARIANT': '0'}
    eval_env = {**os.environ, 'VLLM_BATCH_INVARIANT': '1'}

    def evaluate(step, model):
        with phase(logs, 'bare_student_benchmark_evaluation', step=step):
            subprocess.run([*command, '--eval-worker', str(frozen), '--step', str(step), '--model', str(model)],
                           env=eval_env, check=True)

    with ResourceMonitor(logs, admission['gpu_ids'], tensorboard_dir=logs / 'tensorboard/system'):
        if not evaluation_complete(plan, 0):
            evaluate(0, student)
        latest = recover_checkpoint(checkpoints, plan)
        if latest is None or latest.name != f'global_step_{steps[-1]}':
            train_env['TENSORBOARD_DIR'] = str(logs / 'tensorboard/train')
            with phase(logs, 'sft_training'):
                subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=2',
                                '--module', ENTRY, '--train-worker', str(frozen)], env=train_env, check=True)
            latest = recover_checkpoint(checkpoints, plan)
            if latest is None or latest.name != f'global_step_{steps[-1]}':
                raise RuntimeError('SFT returned without the final complete checkpoint')
        for step in steps[1:]:
            if evaluation_complete(plan, step):
                continue
            if responses_complete(run, step):
                evaluate(step, student)
                continue
            folder = checkpoints / f'global_step_{step}'
            if folder != latest:
                validate_model_checkpoint(folder, plan)
            with TemporaryDirectory(prefix='.sft_hf_', dir=run / 'evaluation') as model:
                subprocess.run([sys.executable, '-m', 'verl.model_merger', 'merge', '--backend', 'fsdp',
                                '--local_dir', str(folder), '--target_dir', model, '--use_cpu_initialization'],
                               env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}, check=True)
                evaluate(step, model)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--execute-prepared', type=Path, help=argparse.SUPPRESS)
    mode.add_argument('--train-worker', type=Path, help=argparse.SUPPRESS)
    mode.add_argument('--eval-worker', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--step', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--model', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--config', default='configs/ablation.json')
    parser.add_argument('--teacher-dir', default=DEFAULT_TEACHER_DIR)
    parser.add_argument('--run-name', default='sft_n1024_seed42_v1')
    parser.add_argument('--train-output-root')
    parser.add_argument('--gpu-ids', default='2,3')
    parser.add_argument('--wait-seconds', type=int, default=3600)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--total-steps', type=int, help='Split one epoch into this many updates')
    parser.add_argument('--save-freq', type=int, default=16)
    parser.add_argument('--presence-penalty', type=float, default=DEFAULT_PRESENCE_PENALTY)
    args = parser.parse_args()
    frozen = args.execute_prepared or args.train_worker or args.eval_worker
    if frozen:
        plan = read_json(frozen)
        validate_prepared(plan)
        admission = admitted_pool(plan, os.environ.get('CUDA_VISIBLE_DEVICES', ''),
                                  os.environ.get('STEER_OPD_GPU_ADMISSION'))
        activate_verl()
        if args.train_worker:
            train_worker(plan)
        elif args.eval_worker:
            evaluate_worker(plan, args.step, args.model)
        else:
            execute(plan, frozen, admission)
        return
    plan, rows, benchmarks = build_plan(args)
    activate_verl()
    config = sft_config(plan)
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if not (args.prepare or args.run):
        return
    run = Path(plan['output'])
    (run / 'metadata').mkdir(parents=True, exist_ok=True)
    with (run / 'metadata/sft_launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        frozen = prepare(plan, rows, benchmarks, config)
        if args.run:
            latest = recover_checkpoint(run / 'checkpoints', plan)
            steps = plan['evaluation']['steps']
            if (latest is not None and latest.name == f'global_step_{steps[-1]}'
                    and all(evaluation_complete(plan, step) for step in steps)):
                print(f'SFT baseline already complete: {args.run_name}', flush=True)
                return
            require_training_space(run, (48 if latest else 96) * 1024**3)
            session = run / 'shared_gpu_admission' / ('launch_' + uuid.uuid4().hex[:12])
            subprocess.run([sys.executable, '-m', 'steer_opd.training.idle_launch', '--gpu-ids', args.gpu_ids,
                            '--wait-seconds', str(args.wait_seconds), '--run-dir', str(session), '--',
                            sys.executable, '-m', ENTRY, '--execute-prepared', str(frozen)],
                           cwd=ROOT, env=child_environment(sys.executable), check=True)


if __name__ == '__main__':
    main()
