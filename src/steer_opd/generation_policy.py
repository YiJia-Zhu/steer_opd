"""Shared defaults for future generation, distinct from raw probability scores."""
import math
from numbers import Real

DEFAULT_PRESENCE_PENALTY = 2.0
DEFAULT_MAX_GENERATION_LENGTH = 16384
PRESENCE_FIELD = 'steer_opd_presence_penalty'
POLICY_VERSION = 'presence_controlled_generation_raw_model_scores_v1'
RAW_PROBABILITY_BASIS = 'raw_model_logprobs_before_sampling_transforms'


def presence_penalty(config=None, *, require_explicit=False):
    config = {} if config is None else config
    if require_explicit and 'presence_penalty' not in config:
        raise ValueError('Prepare a new generation job with explicit presence_penalty')
    value = config.get('presence_penalty', DEFAULT_PRESENCE_PENALTY)
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not -2 <= value <= 2:
        raise ValueError('presence_penalty must be a finite number in [-2, 2]')
    return float(value)


def training_sampling_policy(config):
    penalty = presence_penalty(config)
    return {'version': POLICY_VERSION, 'presence_penalty': penalty,
            'probability_basis': RAW_PROBABILITY_BASIS,
            'raw_logprobs_are_behavior_logprobs': penalty == 0.,
            'loss': 'sampled_raw_model_opd_on_controlled_rollouts',
            'importance_correction': 'none',
            'note': 'Penalty changes the sampling distribution. Raw model logprobs remain the OPD/Top16 inputs; no exact unpenalized on-policy KL equivalence is claimed.'}


def validate_training_sampling_policy(plan):
    settings = plan.get('settings', {})
    if ('presence_penalty' not in settings or
            plan.get('sampling_policy') != training_sampling_policy(settings)):
        raise ValueError('Prepare a new training plan with explicit presence_penalty and raw-model probability semantics')
    return plan['sampling_policy']
