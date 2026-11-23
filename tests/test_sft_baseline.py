"""Verified teacher supervision, answer masks, and the native SFT save layout."""
import hashlib
import importlib
from pathlib import Path

import pytest

from steer_opd.experiment_runtime import StageCheckpoint
from steer_opd.resumable_core import atomic_json, digest
from steer_opd.retention import ALL_MODELS_LATEST_STATE, MINIMAL
from steer_opd.sft_baseline import (
    TokenizedAnswers, encode_example, filter_correct_teacher_answers, one_epoch_batches, teacher_answers, write_parquet,
)
from steer_opd.training.retention import (
    SFT_LAYOUT, commit_checkpoint, prune_checkpoints, recover_checkpoint, validate_model_checkpoint,
)


def test_teacher_source_requires_verified_normal_nonempty_answers(tmp_path):
    questions = {'a': 'question A', 'b': 'question B', 'c': 'question C'}
    teacher = 'teacher'
    requests = [{'id': qid, 'key': qid, 'question': question, 'prompt_token_ids': [i]}
                for i, (qid, question) in enumerate(questions.items())]
    job = {'role': 'teacher', 'model_content_id': 'teacher-content', 'requests': requests}
    identity = digest(job)
    record = tmp_path / 'job.json'
    atomic_json(record, {'job': job, 'hash': identity})
    results = [record]
    for request, (text, reason) in zip(requests, [('derivation</think>final', 'stop'),
                                                   ('unfinished', 'length'), (' ', 'stop')], strict=True):
        path = tmp_path / f"{request['key']}.json"
        atomic_json(path, {**request, 'text': text, 'finish_reason': reason, 'steering': False,
                           'model': teacher, 'model_content_id': 'teacher-content', 'job_hash': identity})
        results.append(path)
    StageCheckpoint(tmp_path, 'test').commit(results, requests=3)
    assert teacher_answers(tmp_path, questions, teacher, 'teacher-content') == (
        {'a': 'derivation</think>final'}, identity,
    )
    with pytest.raises(ValueError, match='frozen training pool'):
        teacher_answers(tmp_path, {'a': questions['a'], 'b': questions['b'], 'other': questions['c']},
                        teacher, 'teacher-content')
    (tmp_path / 'a.json').write_text('{}')
    with pytest.raises(ValueError, match='Committed stage file'):
        teacher_answers(tmp_path, questions, teacher, 'teacher-content')


def test_student_prompt_is_masked_and_response_is_encoded_in_full(tmp_path):
    class Tokenizer:
        eos_token_id = 9

        def apply_chat_template(self, messages, **options):
            assert messages[0]['content'].startswith('question ')
            assert options['enable_thinking'] is False
            assert options['add_generation_prompt'] is True
            return [1, 2, 3]

        def encode(self, text, *, add_special_tokens):
            assert text == 'derivation</think>final'
            assert add_special_tokens is False
            return [4, 5, 6]

    path = tmp_path / 'train.parquet'
    write_parquet(path, [encode_example(Tokenizer(), 'qid', 'question', 'derivation</think>final')])
    entry = TokenizedAnswers(path)[0]
    assert entry['input_ids'].tolist() == [1, 2, 3, 4, 5, 6, 9]
    assert entry['loss_mask'].tolist() == [0, 0, 0, 1, 1, 1, 1]


def test_correctness_screen_uses_final_box_and_requires_reference(monkeypatch):
    from verl.utils.reward_score import math_verify

    questions = {'correct': 'A', 'revised_to_wrong': 'B', 'missing_reference': 'C', 'unclosed': 'D', 'empty': 'E'}
    answers = {
        'correct': r'Derivation \boxed{\frac{1}{2}}.',
        'revised_to_wrong': r'Draft \boxed{1/2}. Final \boxed{2}.',
        'missing_reference': r'\boxed{0}',
        'unclosed': r'Draft \boxed{1/2}. Final \boxed{2',
        'empty': r'Draft \boxed{1/2}. Final \boxed{}',
    }
    references = {hashlib.sha256(text.encode()).hexdigest(): '1/2'
                  for qid, text in questions.items() if qid != 'missing_reference'}
    seen = []

    def grade(final, expected, timeout):
        seen.append((final, expected))
        return float(final == r'\boxed{\frac{1}{2}}' and expected == '1/2')

    monkeypatch.setattr(math_verify, 'compute_score', grade)
    retained, counts = filter_correct_teacher_answers(answers, questions, references)
    assert retained == {'correct': answers['correct']}
    assert seen == [(r'\boxed{\frac{1}{2}}', '1/2'), (r'\boxed{2}', '1/2')]
    assert counts == {'referenced_questions': 4, 'normal_nonempty': 5, 'missing_reference': 1,
                      'missing_final_box': 2, 'rejected_by_grader': 1, 'accepted': 1}


def test_sft_step_override_splits_tail_and_keeps_one_pass():
    assert one_epoch_batches(974, 32, 2) == [32] * 30 + [14]
    assert one_epoch_batches(974, 32, 2, 32) == [32] * 30 + [8, 6]
    with pytest.raises(ValueError, match='supports 31'):
        one_epoch_batches(974, 32, 2, 30)


def test_sft_expansion_skips_questions_with_conflicting_references(monkeypatch):
    import pyarrow as pa

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    expansion = importlib.import_module('prepare_sft_expansion')
    table = pa.table({'problem': ['ambiguous', 'valid', 'ambiguous', ' valid '],
                      'answer': ['1', '7', '2', '7'],
                      'problem_source': ['math'] * 4})

    class Parquet:
        def iter_batches(self, *, columns):
            yield from table.select(columns).to_batches()

    answers, conflicts = expansion.published_answers(Parquet(), 'answer', {'ambiguous', 'valid'})
    assert answers == {'valid': ('7', 'math')}
    assert conflicts == 1


def test_sft_checkpoint_keeps_earlier_shards_and_only_latest_resume_state(tmp_path):
    plan = {'binding': 'sft', 'checkpoint_layout': SFT_LAYOUT, 'settings': {'gpu_pool': [4, 5]},
            'retention_policy': MINIMAL, 'checkpoint_retention': ALL_MODELS_LATEST_STATE}
    tracker = tmp_path / 'latest_checkpointed_iteration.txt'
    for step in (16, 31):
        folder = tmp_path / f'global_step_{step}'
        folder.mkdir()
        for rank in range(2):
            for kind in ('model', 'optim', 'extra_state'):
                (folder / f'{kind}_world_size_2_rank_{rank}.pt').write_bytes(b'checkpoint shard')
            (folder / f'data_{rank}.pt').write_bytes(b'dataloader')
        tracker.write_text(str(step))
        receipt = commit_checkpoint(tmp_path, step, plan)
        prune_checkpoints(tmp_path, step, plan, committed=receipt)
    assert recover_checkpoint(tmp_path, plan) == tmp_path / 'global_step_31'
    old = tmp_path / 'global_step_16'
    assert set(validate_model_checkpoint(old, plan)['files']) == {
        'model_world_size_2_rank_0.pt', 'model_world_size_2_rank_1.pt',
    }
    assert not list(old.glob('data_*.pt')) and not list(old.glob('optim_*.pt'))
