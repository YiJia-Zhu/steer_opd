from importlib import util
import inspect
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from steer_opd.evaluation_text import FORMAT, response_path, responses_complete, save_responses
from steer_opd.resumable_core import ROOT


def freeze_questions(run, questions):
    (run / 'data').mkdir()
    rows = [{'extra_info': {'question_id': qid, 'question': q}} for qid, q in questions.items()]
    pq.write_table(pa.Table.from_pylist(rows), run / 'data/benchmarks.parquet')


def test_archive_preserves_text_and_a_failed_retry_leaves_the_previous_file(tmp_path):
    ids = ['b:1', 'a:0']
    questions = {'a:0': '第一道题\n第二行', 'b:1': 'What is $1+1$?'}
    texts = [' <think>\n' + '中文🙂\\boxed{2}\n' * 16384 + '</think>  ', '']
    freeze_questions(tmp_path, questions)
    save_responses(tmp_path, 0, ids, texts)
    target = response_path(tmp_path, 0)
    original = target.read_bytes()
    assert [json.loads(line) for line in original.decode().splitlines()] == [
        {'question_id': q, 'question': questions[q], 'text': t} for q, t in zip(ids, texts)]
    for questions, bad in [(ids[:1], texts[:1]), (ids[:1] * 2, texts), (ids, ['ok', None])]:
        with pytest.raises(ValueError):
            save_responses(tmp_path, 0, questions, bad)
        assert target.read_bytes() == original
    assert list(target.parent.iterdir()) == [target]
    assert responses_complete(tmp_path, 0)
    target.write_bytes(original.split(b'\n', 1)[0] + b'\n')
    assert not responses_complete(tmp_path, 0)


def runtime_module(src, name):
    spec = util.spec_from_file_location(name, src / 'steer_opd/training' / f'{name}.py')
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('variant', [[], ['--baseline']], ids=['steer', 'baseline'])
def test_real_verl_validation_saves_every_unpadded_answer_once(tmp_path, monkeypatch, variant):
    from steer_opd.training.benchmarks import benchmark_rows
    from steer_opd.training.verl_entry import activate_verl
    activate_verl()
    from omegaconf import OmegaConf
    from verl import DataProto
    from verl.utils.reward_score import math_verify

    src = Path(subprocess.check_output([sys.executable, str(ROOT / 'scripts/prepare_training_runtime.py'),
                                        *variant], text=True).strip())
    reward = runtime_module(src, 'verl_reward')
    # Get the project's actual trainer without starting a Ray worker or loading model weights.
    actor = runtime_module(src, 'verl_trainer').task_runner_class()
    run_method = inspect.unwrap(actor.__ray_metadata__.modified_class.run)
    trainer_type = inspect.getclosurevars(run_method).nonlocals['SteerTrainer']
    trainer = object.__new__(trainer_type)
    rows, _ = benchmark_rows([])
    questions = {r['extra_info']['question_id']: r['extra_info']['question'] for r in rows}
    ids = list(questions)
    freeze_questions(tmp_path, questions)
    index = {qid: i+1 for i, qid in enumerate(ids)}
    answers = {i: f' <think>推导 {i}\n</think>\n\\boxed{{{i}}}  ' for i in index.values()}
    monkeypatch.setattr(math_verify, 'compute_score', lambda *args, **kwargs: 1.)
    generated_sizes = []

    def generate(batch):
        infos = batch.non_tensor_batch['extra_info']
        codes = [index[info['question_id']] for info in infos]
        generated_sizes.append(len(infos))
        grades = [reward.compute_score(source, answers[i], '1', extra_info=info)
                  for source, i, info in zip(batch.non_tensor_batch['data_source'], codes, infos)]
        return DataProto.from_dict(
            tensors={'responses': torch.tensor(codes)[:, None], 'rm_scores': torch.ones(len(codes), 1)},
            non_tensors={key: np.array([grade[key] for grade in grades]) for key in grades[0]},
            meta_info={'reward_extra_keys': list(grades[0])})

    def inputs(group):
        return {'prompts': torch.zeros(len(group), 1, dtype=torch.long),
                **{k: np.array([row[k] for row in group], dtype=object)
                   for k in ('extra_info', 'data_source', 'reward_model')}}

    trainer.steer_plan = {'output': str(tmp_path), 'evaluation_response_format': FORMAT}
    trainer.global_steps = 16
    trainer.log_folder = tmp_path / 'logs'
    trainer.handover = SimpleNamespace(require_student=lambda: None)
    trainer.use_rm = False
    trainer.tokenizer = SimpleNamespace(eos_token_id=0, pad_token_id=0,
        decode=lambda tokens, skip_special_tokens: answers.get(int(tokens[0]), 'prompt'))
    trainer.async_rollout_manager = SimpleNamespace(generate_sequences=generate)
    # Both batches need padding. Deliberately change the order relative to the frozen parquet.
    rows.reverse()
    trainer.val_dataloader = [inputs(rows[:301]), inputs(rows[301:])]
    trainer.config = OmegaConf.create({
        'actor_rollout_ref': {'rollout': {'agent': {'num_workers': 8}, 'val_kwargs': {'n': 1, 'do_sample': True}}},
        'trainer': {'log_val_generations': 0, 'validation_data_dir': str(response_path(tmp_path, 16).parent)}})
    metrics = trainer._validate()
    assert generated_sizes == [304, 232]
    assert metrics['eval/AIME2025/acc'] == metrics['eval/MATH-500/acc'] == 1.
    assert all('question_id' not in key for key in metrics)
    assert json.loads((tmp_path / 'evaluation/metrics/step_16.json').read_text()) == metrics
    records = [json.loads(line) for line in response_path(tmp_path, 16).read_text().splitlines()]
    assert records == [{'question_id': q, 'question': questions[q], 'text': answers[index[q]]}
                       for q in reversed(ids)]
    assert responses_complete(tmp_path, 16)
