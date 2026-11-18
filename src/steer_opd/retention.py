"""Versioned retention choices; missing policies preserve frozen historical runs."""

MINIMAL = 'minimal_v1'
FULL = 'full_v1'


def minimal(value):
    policy = value.get('retention_policy')
    if policy not in (None, MINIMAL, FULL):
        raise ValueError('Unknown retention policy')
    return policy == MINIMAL
