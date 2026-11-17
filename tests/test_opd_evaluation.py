"""Evaluation count, aggregation and extraction must survive partial/invalid output."""
from types import SimpleNamespace as NS

import pytest

from steer_opd.opd_evaluation import aime_questions, grade_answer, pack_outputs, score_outputs
from steer_opd.resumable_core import ROOT


def outputs():
    return [{"question_id": question, "sample_index": sample, "text": "yes" if (question == "a" and sample < 2) else "no"}
            for question in ("a", "b") for sample in range(16)]


def grader(text, gold):
    return {"correct": text == "yes", "answer_extracted": True, "extracted_answer": [text]}


def test_acc_and_pass16_have_different_denominators():
    metrics, per_question, _ = score_outputs([{"id": "a", "answer": "0"}, {"id": "b", "answer": "1"}], outputs(), grader=grader)
    assert metrics["acc"] == 2 / 32
    assert metrics["pass@16"] == 1 / 2
    assert per_question[0]["acc"] == 2 / 16


@pytest.mark.parametrize("change", ["missing", "duplicate", "foreign", "index", "bool_index"])
def test_partial_or_duplicate_evaluations_do_not_produce_metrics(change):
    rows = outputs()
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows[-1] = dict(rows[0])
    elif change == "foreign":
        rows[-1]["question_id"] = "c"
    elif change == "index":
        rows[-1]["sample_index"] = 16
    else:
        rows[-1]["sample_index"] = True
    with pytest.raises(ValueError):
        score_outputs([{"id": "a", "answer": "0"}, {"id": "b", "answer": "1"}], rows, grader=grader)


def test_existing_math_verifier_extracts_boxed_gold_and_rejects_empty_answer():
    assert grade_answer(r"The answer is $\boxed{070}$.", "70")["correct"]
    assert not grade_answer(r"The answer is $\boxed{71}$.", "70")["correct"]
    assert not grade_answer("", "70")["correct"]


def test_local_aime_has30_gold_questions():
    questions, sources = aime_questions(ROOT / "assets/data/AIME2025")
    assert len(questions) == 30 and len(sources) == 2


def test_pack_preserves_text_tokens_and_finish_reason():
    candidate = NS(index=0, text="text", token_ids=[5, 6], cumulative_logprob=-1.2,
                   finish_reason="length", stop_reason=None)
    result = NS(prompt_token_ids=[1, 2], outputs=[candidate], request_id="req")
    rows = pack_outputs([{"id": "q"}], [[1, 2]], [result], n=1)
    assert rows[0]["token_ids"] == [5, 6] and rows[0]["text"] == "text"
    assert rows[0]["finish_reason"] == "length"
    result.outputs = [candidate, candidate]
    with pytest.raises(ValueError):
        pack_outputs([{"id": "q"}], [[1, 2]], [result], n=2)
