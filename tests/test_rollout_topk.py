"""Raw Top-K ranks, observed-token extras and chunked reference projections."""
from types import SimpleNamespace
import pytest
import torch
from steer_opd.resumable_core import score_hidden
from steer_opd.rollout_topk import extract_topk


def entries(values, ranks):
    return {key: SimpleNamespace(logprob=value, rank=rank)
            for (key, value), rank in zip(values, ranks)}


def sample_score():
    return extract_topk([4, 1], [entries([(4, -4), (0, -0.5), (1, -1.5), (2, -2.5)], [5, 1, 2, 3]),
                                entries([(2, -2.5), (1, -1.5), (0, -0.5)], [3, 2, 1])], 3, 7)


def test_extra_sample_is_excluded_even_when_its_rank_ties_topk_boundary():
    score = sample_score()
    assert score["ids"].tolist() == [[0, 1, 2], [0, 1, 2]]
    assert score["sampled_logps"].tolist() == [-4, -1.5]
    tied = extract_topk([6], [entries([(6, -2.5), (0, -0.5), (1, -1.5), (2, -2.5)], [3, 1, 2, 3])], 3, 7)
    assert tied["ids"].tolist() == [[0, 1, 2]]


@pytest.mark.parametrize("steps", [None, [], [entries([(0, -1)], [None])],
                                   [entries([(0, -1), (1, -2)], [1, 3])],
                                   [entries([(0, float('nan')), (1, -2)], [1, 2])],
                                   [entries([(0, -1), (9, -2)], [1, 2])]])
def test_incomplete_nonfinite_or_invalid_candidates_rejected(steps):
    with pytest.raises(ValueError):
        extract_topk([0], steps, 2, 7)


def test_teacher_candidates_and_own_topk_use_one_projection_per_chunk(monkeypatch):
    import steer_opd.resumable_core as core
    model = SimpleNamespace(lm_head=torch.nn.Linear(4, 7, bias=False),
                            config=SimpleNamespace(vocab_size=7), _opd_vocab_size=7)
    hidden = torch.randn(5, 4)
    ids = torch.tensor([[6, 0, 3]] * 5)
    calls = []
    original = core.project_logits

    def record(*args, **kw):
        calls.append(True)
        return original(*args, **kw)

    monkeypatch.setattr(core, "project_logits", record)
    with torch.no_grad():
        score = score_hidden(model, hidden, k=3, candidates=ids, chunk=2, own_topk=True)
    assert len(calls) == 3
    lp = original(model, hidden).log_softmax(-1)
    torch.testing.assert_close(score["logps"], lp.gather(-1, ids))
    torch.testing.assert_close(score["own_logps"], lp.topk(3).values)
    assert torch.equal(score["own_ids"].long(), lp.topk(3).indices)
