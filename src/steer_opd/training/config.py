"""Prospective OPD loss settings, independent of the frozen overlap campaign."""
from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
from typing import Mapping

from steer_opd.resumable_core import ROOT

ALIGNMENT = 'compass_a4_shared_text_span_mean_native_logp_v1'
DEFAULT_REFERENCE = 'assets/models/Qwen3.5-0.8B'


@dataclass(frozen=True)
class TrainingLossConfig:
    loss_type: str = 'traditional_opd'
    anchor_coef: float = 0.5
    reference_model: str = DEFAULT_REFERENCE
    alignment: str = ALIGNMENT

    def __post_init__(self):
        if self.loss_type not in ('traditional_opd', 'compass_opd'):
            raise ValueError('loss_type must be traditional_opd or compass_opd')
        if (isinstance(self.anchor_coef, bool) or not isinstance(self.anchor_coef, (int, float))
                or not math.isfinite(self.anchor_coef) or self.anchor_coef < 0):
            raise ValueError('anchor_coef must be finite and nonnegative')
        if not isinstance(self.reference_model, str) or not self.reference_model.strip():
            raise ValueError('Set a teacher-family reference checkpoint')
        if self.alignment != ALIGNMENT:
            raise ValueError('Both losses use the same CompassOPD Appendix A.4 text alignment')

    @classmethod
    def from_experiment(cls, experiment: Mapping):
        """Read optional training_loss; absent settings select conventional S/T OPD."""
        return cls(**experiment.get('training_loss', {}))


def training_plan(experiment: Mapping, settings=None):
    """Declare required roles without importing a model backend or changing holds.

    The frozen student anchor is a snapshot of the initial student, not R and
    not the moving student checkpoint after updates. This is a CPU-only plan;
    the existing overlap runner intentionally does not call it.
    """
    settings = settings or TrainingLossConfig.from_experiment(experiment)
    roles = {
        'student': {'checkpoint': experiment['student'], 'trainable': True},
        'teacher': {'checkpoint': experiment['teacher'], 'trainable': False},
    }
    if settings.loss_type == 'compass_opd':
        roles['reference'] = {'checkpoint': settings.reference_model, 'trainable': False,
                              'family_requirement': 'same_release_family_as_teacher',
                              'steering': False}
        roles['student_anchor'] = {'initial_checkpoint': experiment['student'], 'trainable': False,
                                   'snapshot_before_first_update': True, 'steering': False}
    sources = [*Path(__file__).resolve().parent.glob('*.py'), ROOT/'src/steer_opd/compass_alignment.py']
    return {
        'stage': 'opd_loss_preparation', 'training_started': False,
        'settings': asdict(settings), 'model_roles': roles,
        'loss_reduction': 'mean_over_valid_student_tokens',
        'rollout_policy': 'current_student_on_policy',
        'student_logps': 'fresh_differentiable_forward_on_actual_rollout',
        'teacher_logps': 'observed_native_tokens_of_the_same_student_response',
        'reference_checkpoint_validation': 'required_before_training' if settings.loss_type == 'compass_opd' else 'unused',
        'execution_hold': experiment.get('execution_hold'),
        'implementation_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted(sources)},
    }
