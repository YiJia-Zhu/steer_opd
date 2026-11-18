"""Span pooling preserves OPD values, masks and gradients across tokenizations."""
from copy import deepcopy

import pytest
import torch

from steer_opd.resumable_core import digest
from steer_opd.training.config import TrainingLossConfig
from steer_opd.training.losses import _span_means, distillation_loss


def irregular_inputs(kind, dtype):
    generator = torch.Generator().manual_seed(42)
    roles = ['student', 'teacher'] + (['reference'] if kind == 'compass_opd' else [])
    sizes = torch.randint(1, 8, (160, len(roles)), generator=generator).tolist()
    positions = dict.fromkeys(roles, 0)
    units, mask = [], []
    for index, lengths in enumerate(sizes):
        ranges = {}
        for role, length in zip(roles, lengths):
            start = positions[role]
            positions[role] += length
            ranges[role] = [start, positions[role]]
        valid = index % 13 != 5
        units.append({'ranges': ranges, 'valid': valid})
        mask.extend([valid] * lengths[0])
    plan = {'source_token_ids': list(range(positions['student'])) + [9999],
            'native_response_ids': {role: list(range(n)) for role, n in positions.items()},
            'units': units, 'alignment_mask': mask + [False]}
    plan['alignment_id'] = digest(plan)
    scores = {}
    for role, count in positions.items():
        count += role == 'student'
        # Noncontiguous scores also occur when selecting a padded batch row.
        values = (-.01 - 6 * torch.rand(count * 2, generator=generator, dtype=dtype))[::2]
        for unit in units:
            if not unit['valid']:
                a, b = unit['ranges'][role]
                values[a:b] = float('nan')
        if role == 'student':
            values[-1] = float('nan')
        scores[role] = values.requires_grad_()
    return plan, scores


def scalar_span_means(plan, values, role, output):
    """Original per-span definition, used only as an independent CPU oracle."""
    for unit in plan['units']:
        if unit['valid']:
            a, b = unit['ranges']['student']
            start, end = unit['ranges'][role]
            output[a:b] = values[start:end].detach().to(output).mean()
    return output


@pytest.mark.parametrize('kind', ['traditional_opd', 'compass_opd'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_irregular_spans_match_scalar_advantages_loss_and_student_gradient(kind, dtype):
    plan, scores = irregular_inputs(kind, dtype)
    student = scores['student']
    settings = TrainingLossConfig(loss_type=kind, anchor_coef=.5)
    response_mask = torch.arange(len(student)) % 7 != 0
    mask = torch.tensor(plan['alignment_mask']) & response_mask
    teacher_mean = scalar_span_means(plan, scores['teacher'], 'teacher', torch.zeros_like(student))
    kwargs = {}
    if kind == 'traditional_opd':
        other_mean = scalar_span_means(plan, student, 'student', torch.zeros_like(student))
        expected = teacher_mean - other_mean
    else:
        anchor = (student.detach() - .25).requires_grad_()
        kwargs = {'reference_logps': scores['reference'], 'student_anchor_logps': anchor}
        other_mean = scalar_span_means(plan, scores['reference'], 'reference', torch.zeros_like(student))
        expected = teacher_mean - other_mean - .5 * (student.detach() - anchor.detach())
    expected = expected.masked_fill(~mask, 0.)
    result = distillation_loss(plan, student, scores['teacher'], settings=settings,
                               response_mask=response_mask, **kwargs)
    tolerance = 2e-6 if dtype == torch.float32 else 1e-12
    torch.testing.assert_close(result.advantages, expected, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(result.alignment_mask, mask)
    expected_loss = -(expected[mask] * student.detach()[mask]).mean()
    torch.testing.assert_close(result.loss, expected_loss, atol=tolerance, rtol=tolerance)
    result.loss.backward()
    expected_gradient = -expected / int(mask.sum())
    torch.testing.assert_close(student.grad, expected_gradient, atol=tolerance, rtol=tolerance)
    assert result.valid_tokens == int(mask.sum())
    assert not result.advantages.requires_grad
    assert all(value.grad is None for role, value in scores.items() if role != 'student')
    if kind == 'compass_opd':
        assert anchor.grad is None


@pytest.mark.parametrize('bad_score', [float('nan'), float('inf'), .1])
def test_nonfinite_or_positive_scores_inside_valid_spans_are_rejected(bad_score):
    plan, scores = irregular_inputs('traditional_opd', torch.float32)
    teacher = scores['teacher'].detach().clone()
    teacher[0] = bad_score
    with pytest.raises(ValueError, match='nonfinite or positive'):
        distillation_loss(plan, scores['student'], teacher)


@pytest.mark.parametrize('role,bounds', [('student', [-1, 1]), ('teacher', [0, 99999])])
def test_invalid_span_bounds_are_still_rejected(role, bounds):
    plan, scores = irregular_inputs('traditional_opd', torch.float32)
    plan = deepcopy(plan)
    plan['units'][0]['ranges'][role] = bounds
    plan['alignment_id'] = digest({k: v for k, v in plan.items() if k != 'alignment_id'})
    with pytest.raises(ValueError, match='outside native logprob positions'):
        distillation_loss(plan, scores['student'], scores['teacher'])


def test_small_fp32_scores_after_long_span_do_not_lose_precision_to_prefix_sums():
    values = torch.full((7168,), -100.)
    values[-2:] = torch.tensor([-1e-8, -.00234567])
    plan = {'units': [
        {'valid': True, 'ranges': {'student': [0, 1], 'teacher': [0, 7166]}},
        {'valid': True, 'ranges': {'student': [1, 2], 'teacher': [7166, 7167]}},
        {'valid': True, 'ranges': {'student': [2, 3], 'teacher': [7167, 7168]}},
    ]}
    actual = _span_means(plan, values, 'teacher', torch.zeros(4))
    torch.testing.assert_close(actual, torch.tensor([-100., -1e-8, -.00234567, 0.]),
                               rtol=1e-6, atol=1e-12)


def test_no_valid_spans_leave_output_untouched():
    output = torch.tensor([7., 8.])
    plan = {'units': [{'valid': False, 'ranges': {'student': [-1, 20], 'teacher': [-1, 20]}}]}
    torch.testing.assert_close(_span_means(plan, torch.tensor([float('nan')]), 'teacher', output),
                               torch.tensor([7., 8.]))
