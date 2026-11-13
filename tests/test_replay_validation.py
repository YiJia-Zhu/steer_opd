"""Short rare-token errors need evidence from the actual sampled distribution."""

import pytest
import torch

from steer_opd.replay_validation import verify_short_replay


def test_small_distribution_change_can_have_large_rare_token_logp_error():
    captured = torch.tensor([[3.0, 2.0, -5.0], [3.0, 2.0, -5.0]])
    replay = captured.clone()
    replay[:, -1] += 0.3
    sampled = captured.log_softmax(-1)[:, -1]
    assert (replay.log_softmax(-1)[:, -1] - sampled).abs().mean() > 0.12
    result = verify_short_replay(captured, replay, [2, 2], sampled)
    assert result["passed"]
    assert max(result["forward_kls"]) < 0.005


def test_wrong_distribution_is_rejected():
    captured = torch.tensor([[3.0, 0.0]])
    replay = torch.tensor([[0.0, 3.0]])
    with pytest.raises(RuntimeError, match="full-distribution mismatch"):
        verify_short_replay(captured, replay, [0], captured.log_softmax(-1)[:, 0])


def test_capture_from_another_rollout_is_rejected():
    captured = torch.tensor([[3.0, 0.0]])
    with pytest.raises(RuntimeError, match="full-distribution mismatch"):
        verify_short_replay(captured, captured, [0], [-1.0])


def test_incomplete_long_or_nonfinite_distributions_are_rejected():
    with pytest.raises(RuntimeError, match="shape mismatch"):
        verify_short_replay(torch.zeros(2, 3), torch.zeros(1, 3), [0, 0], [-1.0, -1.0])
    with pytest.raises(RuntimeError, match="complete short replies"):
        verify_short_replay(torch.zeros(9, 3), torch.zeros(9, 3), [0] * 9, [-1.0] * 9)
    with pytest.raises(RuntimeError, match="full-distribution mismatch"):
        verify_short_replay(torch.tensor([[float('nan'), 0.0]]), torch.zeros(1, 2), [0], [-1.0])
