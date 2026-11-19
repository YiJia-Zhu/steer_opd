"""Differentiable sampled OPD losses with CompassOPD text-span alignment.

Source: CompassOPD, arXiv:2609.10154v1, equations (2), (4), (7)-(9)
and Appendix A.4. This is a paper-based implementation, not an import of
unavailable official training code. These kernels do not launch training.
"""
from dataclasses import dataclass
from typing import Mapping

import torch

from steer_opd.compass_alignment import align_response
from steer_opd.resumable_core import digest
from steer_opd.training.config import TrainingLossConfig


def training_alignment(source_ids, student_codec, teacher_codec, *, settings=None, reference_codec=None):
    """Keep actual student IDs; retokenize their exact response bytes per scorer."""
    settings = settings or TrainingLossConfig()
    scorers = {'teacher': teacher_codec}
    if settings.loss_type == 'compass_opd':
        if reference_codec is None:
            raise ValueError('CompassOPD requires the teacher-family reference tokenizer')
        scorers['reference'] = reference_codec
    elif reference_codec is not None:
        raise ValueError('Traditional OPD uses S/T only; do not load Ref')
    return align_response(source_ids, student_codec, scorers, invalid_utf8='mask')


@dataclass
class DistillationLoss:
    loss: torch.Tensor
    loss_sum: torch.Tensor
    valid_tokens: int
    alignment_mask: torch.Tensor
    advantages: torch.Tensor
    teacher_signal: torch.Tensor
    student_displacement: torch.Tensor
    loss_type: str
    alignment_id: str


def _native_logps(values, length, name):
    if (not isinstance(values, torch.Tensor) or values.ndim != 1
            or values.shape[0] != length or not values.is_floating_point()):
        raise ValueError(f'{name} must contain one floating native logprob per actual response token')
    return values


def _valid_logps(values, name):
    if not torch.isfinite(values).all() or (values > 1e-5).any():
        raise ValueError(f'{name} has nonfinite or positive valid logprobs')


def _span_means(plan, logps, role, output):
    """Reduce native spans together, then broadcast to native student positions.

    Keep reductions local to each span: subtracting long FP32 prefix sums
    would lose small logprobs near the end of a trajectory. Invalid/control
    spans and terminal positions remain untouched, including their NaNs.
    """
    ranges = [(*unit['ranges']['student'], *unit['ranges'][role])
              for unit in plan['units'] if unit['valid']]
    if not ranges:
        return output
    if any(not 0 <= a < b <= len(output) or not 0 <= start < end <= len(logps)
           for a, b, start, end in ranges):
        raise ValueError('Alignment span is outside native logprob positions')

    bounds = torch.tensor(ranges, dtype=torch.int64, device=output.device)
    native_lengths = bounds[:, 3] - bounds[:, 2]
    native_count = sum(end-start for _, _, start, end in ranges)
    native_offsets = native_lengths.cumsum(0) - native_lengths
    native_positions = torch.repeat_interleave(bounds[:, 2] - native_offsets,
        native_lengths, output_size=native_count)
    native_positions += torch.arange(native_count, device=output.device)
    scores = logps.detach().to(device=output.device, dtype=output.dtype)[native_positions]
    _valid_logps(scores, role)
    means = torch.segment_reduce(scores, 'mean', lengths=native_lengths)

    student_lengths = bounds[:, 1] - bounds[:, 0]
    student_count = sum(b-a for a, b, _, _ in ranges)
    student_offsets = student_lengths.cumsum(0) - student_lengths
    student_positions = torch.repeat_interleave(bounds[:, 0] - student_offsets,
        student_lengths, output_size=student_count)
    student_positions += torch.arange(student_count, device=output.device)
    output[student_positions] = torch.repeat_interleave(means, student_lengths,
        output_size=student_count)
    return output


def distillation_loss(plan: Mapping, student_logps, teacher_logps, *, settings=None,
                      reference_logps=None, student_anchor_logps=None, response_mask=None):
    """Compute a detached sampled advantage and a native-student PG surrogate.

    Traditional: A_t = sg[mean_span(log T) - mean_span(log S)].
    Compass: A_t = sg[mean_span(log T) - mean_span(log R)
                         - anchor_coef * (log S_t - log S0_t)].
    The student-anchor displacement is tokenwise, never span-averaged.
    Both routes use L = -sum_valid(A_t * log S_t) / n_valid.

    Native observed-token logps are sufficient; Top16 candidates and overlap
    values are not inputs to this loss. Teacher/R/S0 scores are always detached.
    The student logps must come from the actual current-policy trajectory and
    a fresh differentiable forward; inference caches cannot supply gradients.
    """
    settings = settings or TrainingLossConfig()
    identity = {k: v for k, v in plan.items() if k != 'alignment_id'}
    if plan.get('alignment_id') != digest(identity):
        raise ValueError('Alignment plan content identity changed')
    roles = set(plan['native_response_ids'])
    expected = {'student', 'teacher', 'reference'} if settings.loss_type == 'compass_opd' else {'student', 'teacher'}
    if roles != expected:
        raise ValueError('Alignment roles do not match the selected training loss')
    n = len(plan['source_token_ids'])
    student_logps = _native_logps(student_logps, n, 'student')
    teacher_logps = _native_logps(teacher_logps, len(plan['native_response_ids']['teacher']), 'teacher')
    device = student_logps.device
    # FP32 arithmetic matches the saved native logprobs; preserve FP64 for checks.
    dtype = torch.float64 if student_logps.dtype == torch.float64 else torch.float32
    student = student_logps.to(dtype=dtype)
    mask = torch.as_tensor(plan['alignment_mask'], dtype=torch.bool, device=device)
    if mask.shape != (n,):
        raise ValueError('Alignment mask length changed')
    if response_mask is not None:
        if (not isinstance(response_mask, torch.Tensor) or response_mask.shape != (n,)
                or response_mask.dtype != torch.bool):
            raise ValueError('response_mask must be a boolean vector in native student positions')
        mask = mask & response_mask.to(device=device)
    valid_tokens = int(mask.sum())
    if not valid_tokens:
        raise ValueError('No aligned response tokens contribute to this loss')
    _valid_logps(student[mask], 'student')
    template = torch.zeros(n, dtype=dtype, device=device)
    teacher_mean = _span_means(plan, teacher_logps, 'teacher', template.clone())
    displacement = template.clone()
    if settings.loss_type == 'traditional_opd':
        if reference_logps is not None or student_anchor_logps is not None:
            raise ValueError('Traditional OPD requires no Ref or frozen student anchor scores')
        student_mean = _span_means(plan, student, 'student', template.clone())
        teacher_signal = teacher_mean - student_mean
        advantage = teacher_signal
    else:
        if reference_logps is None or student_anchor_logps is None:
            raise ValueError('CompassOPD requires Ref and frozen initial-student logprobs')
        reference_logps = _native_logps(reference_logps, len(plan['native_response_ids']['reference']), 'reference')
        anchor = _native_logps(student_anchor_logps, n, 'student_anchor').detach().to(device=device, dtype=dtype)
        _valid_logps(anchor[mask], 'student_anchor')
        reference_mean = _span_means(plan, reference_logps, 'reference', template.clone())
        teacher_signal = teacher_mean - reference_mean
        # Avoid evaluating NaN EOS/padding in any loss term, including 0 * NaN.
        displacement[mask] = student.detach()[mask] - anchor[mask]
        advantage = teacher_signal - settings.anchor_coef * displacement
    advantage = advantage.detach().masked_fill(~mask, 0.)
    loss_sum = -(advantage[mask] * student[mask]).sum()
    return DistillationLoss(
        loss=loss_sum / valid_tokens, loss_sum=loss_sum, valid_tokens=valid_tokens,
        alignment_mask=mask, advantages=advantage,
        teacher_signal=teacher_signal.detach().masked_fill(~mask, 0.),
        student_displacement=displacement.detach(), loss_type=settings.loss_type,
        alignment_id=plan['alignment_id'],
    )


def token_mean_loss(results):
    """Combine different-length responses without giving short answers extra weight."""
    results = list(results)
    if not results:
        raise ValueError('Cannot reduce an empty training batch')
    count = sum(item.valid_tokens for item in results)
    return torch.stack([item.loss_sum for item in results]).sum() / count
