"""Versioned retention choices; missing policies preserve frozen historical runs."""

MINIMAL = 'minimal_v1'
FULL = 'full_v1'
ALL_MODELS_LATEST_STATE = 'all_models_latest_state_v1'
LATEST_FULL = 'latest_full_v1'
CHECKPOINT_POLICIES = (ALL_MODELS_LATEST_STATE,LATEST_FULL)


def checkpoint_policy(value):
    policy=value.get('checkpoint_retention',LATEST_FULL)
    if policy not in CHECKPOINT_POLICIES:raise ValueError('Unknown checkpoint retention policy')
    return policy


def keep_model_history(value):
    return checkpoint_policy(value)==ALL_MODELS_LATEST_STATE


def minimal(value):
    policy = value.get('retention_policy')
    if policy not in (None, MINIMAL, FULL):
        raise ValueError('Unknown retention policy')
    return policy == MINIMAL
