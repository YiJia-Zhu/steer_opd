"""Paper-defined OPD advantages, tokenizer spans and actual gradient isolation."""
from copy import deepcopy

import pytest
import torch

from steer_opd.resumable_core import ROOT, read_json
from steer_opd.training import (TrainingLossConfig, distillation_loss,
                                token_mean_loss, training_alignment, training_plan)
from test_compass_alignment import ToyCodec


def inputs(kind='traditional_opd'):
    s = ToyCodec({0: b'<eos>', 1: b'a', 2: b'b', 3: b'c'})
    t = ToyCodec({0: b'<eos>', 11: b'ab', 12: b'c'})
    r = ToyCodec({0: b'<eos>', 21: b'ab', 22: b'c'})
    settings = TrainingLossConfig(loss_type=kind)
    plan = training_alignment([1, 2, 3, 0], s, t, settings=settings,
                              reference_codec=r if kind == 'compass_opd' else None)
    student = torch.tensor([-1., -3., -2., float('nan')], requires_grad=True)
    teacher = torch.tensor([-.5, -1.], requires_grad=True)
    return settings, plan, student, teacher


def test_default_uses_only_student_teacher_and_preserves_overlap_hold():
    cfg = read_json(ROOT/'configs/ablation.json')
    before = deepcopy(cfg)
    plan = training_plan(cfg)
    assert plan['settings']['loss_type'] == 'traditional_opd'
    assert set(plan['model_roles']) == {'student', 'teacher'}
    assert plan['execution_hold'] == 'awaiting_user_approval'
    assert plan['training_started'] is False
    assert cfg == before
    assert 'src/steer_opd/training/losses.py' in plan['implementation_sha256']


def test_compass_plan_requires_distinct_teacher_family_ref_and_student_anchor():
    cfg = {'student': 'S', 'teacher': 'T', 'training_loss': {'loss_type': 'compass_opd'}}
    plan = training_plan(cfg)
    assert set(plan['model_roles']) == {'student', 'teacher', 'reference', 'student_anchor'}
    assert plan['model_roles']['reference']['checkpoint'] == 'assets/models/Qwen3.5-0.8B'
    assert plan['model_roles']['student_anchor']['initial_checkpoint'] == 'S'
    assert plan['model_roles']['student_anchor']['snapshot_before_first_update'] is True
    assert all(not v['trainable'] for k, v in plan['model_roles'].items() if k != 'student')
    assert plan['settings']['anchor_coef'] == .5


def test_traditional_opd_uses_span_means_and_stopped_advantage_gradient():
    settings, plan, student, teacher = inputs()
    result = distillation_loss(plan, student, teacher, settings=settings)
    assert plan['units'][0]['ranges'] == {'student': [0, 2], 'teacher': [0, 1]}
    torch.testing.assert_close(result.advantages, torch.tensor([1.5, 1.5, 1., 0.]))
    assert result.valid_tokens == 3
    assert result.loss.item() == pytest.approx(8 / 3)
    result.loss.backward()
    torch.testing.assert_close(student.grad, torch.tensor([-.5, -.5, -1 / 3, 0.]))
    assert teacher.grad is None
    assert not result.advantages.requires_grad
    assert not torch.isnan(result.loss)


def test_compass_teacher_shift_broadcasts_but_anchor_is_native_tokenwise():
    settings, plan, student, teacher = inputs('compass_opd')
    reference = torch.tensor([-1.5, -2.], requires_grad=True)
    anchor = torch.tensor([-2., -2., -2., float('nan')], requires_grad=True)
    result = distillation_loss(plan, student, teacher, settings=settings,
                               reference_logps=reference, student_anchor_logps=anchor)
    torch.testing.assert_close(result.teacher_signal, torch.tensor([1., 1., 1., 0.]))
    torch.testing.assert_close(result.student_displacement, torch.tensor([1., -1., 0., 0.]))
    torch.testing.assert_close(result.advantages, torch.tensor([.5, 1.5, 1., 0.]))
    result.loss.backward()
    torch.testing.assert_close(student.grad, torch.tensor([-1 / 6, -.5, -1 / 3, 0.]))
    assert all(v.grad is None for v in (teacher, reference, anchor))


def test_compass_initial_policy_has_zero_anchor_displacement():
    settings, plan, student, teacher = inputs('compass_opd')
    result = distillation_loss(plan, student, teacher, settings=settings,
                               reference_logps=torch.tensor([-1.5, -2.]),
                               student_anchor_logps=student.detach().clone())
    torch.testing.assert_close(result.advantages, torch.tensor([1., 1., 1., 0.]))
    assert result.student_displacement.count_nonzero() == 0


def test_reference_and_teacher_different_tokenizations_share_common_text_units():
    s = ToyCodec({0: b'<eos>', 1: b'a', 2: b'b', 3: b'c'})
    t = ToyCodec({0: b'<eos>', 11: b'ab', 12: b'c'})
    r = ToyCodec({0: b'<eos>', 21: b'a', 22: b'bc'})
    settings = TrainingLossConfig(loss_type='compass_opd', anchor_coef=0.)
    plan = training_alignment([1, 2, 3, 0], s, t, settings=settings, reference_codec=r)
    assert len(plan['units']) == 1
    assert plan['units'][0]['ranges'] == {'student': [0, 3], 'teacher': [0, 2], 'reference': [0, 2]}
    result = distillation_loss(plan, torch.tensor([-1., -2., -3., -4.]),
        torch.tensor([-1., -3.]), settings=settings,
        reference_logps=torch.tensor([-3., -5.]), student_anchor_logps=torch.tensor([-1., -2., -3., -4.]))
    torch.testing.assert_close(result.advantages, torch.tensor([2., 2., 2., 0.]))


def test_mask_excludes_eos_and_batch_reduction_weights_valid_tokens():
    settings, plan, student, teacher = inputs()
    full = distillation_loss(plan, student, teacher)
    one = distillation_loss(plan, student, teacher, response_mask=torch.tensor([False, False, True, False]))
    assert one.valid_tokens == 1 and one.loss.item() == pytest.approx(2.)
    assert token_mean_loss([full, one]).item() == pytest.approx((8 + 2) / 4)
    assert full.alignment_mask.tolist() == [True, True, True, False]
    with pytest.raises(ValueError, match='No aligned'):
        distillation_loss(plan, student, teacher, response_mask=torch.zeros(4, dtype=torch.bool))


def test_missing_reference_or_anchor_cannot_silently_become_compass():
    settings, plan, student, teacher = inputs('compass_opd')
    with pytest.raises(ValueError, match='Ref and frozen'):
        distillation_loss(plan, student, teacher, settings=settings)
    settings, plan, student, teacher = inputs()
    with pytest.raises(ValueError, match='requires no Ref'):
        distillation_loss(plan, student, teacher, reference_logps=torch.zeros(2))
    with pytest.raises(ValueError, match='roles'):
        distillation_loss(plan, student, teacher, settings=TrainingLossConfig(loss_type='compass_opd'))


def test_bad_shapes_nonfinite_valid_scores_and_changed_identity_are_rejected():
    _, plan, student, teacher = inputs()
    with pytest.raises(ValueError, match='native logprob'):
        distillation_loss(plan, student[:-1], teacher)
    with pytest.raises(ValueError, match='nonfinite'):
        distillation_loss(plan, student, torch.tensor([float('nan'), -1.]))
    bad = deepcopy(plan)
    bad['alignment_mask'][0] = False
    with pytest.raises(ValueError, match='identity'):
        distillation_loss(bad, student, teacher)
    with pytest.raises(ValueError, match='boolean'):
        distillation_loss(plan, student, teacher, response_mask=torch.ones(4))


@pytest.mark.parametrize('coefficient', [-1., float('nan'), float('inf'), True])
def test_invalid_anchor_coefficients(coefficient):
    with pytest.raises(ValueError, match='anchor_coef'):
        TrainingLossConfig(anchor_coef=coefficient)
