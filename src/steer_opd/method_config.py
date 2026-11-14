"""Validated axes for the frozen-student eight-scheme screen."""
from dataclasses import asdict, dataclass
import math

from steer_opd.operators import validate_operator


DIRECTION_ROLES = {'T-S': ('teacher', 'student'), 'T-R': ('teacher', 'reference')}


@dataclass(frozen=True)
class Method:
    direction: str = 'T-S'
    granularity: str = 'global'
    extractor: str = 'initial'
    layers: tuple = (26,)
    rho: float = .03
    gate: str = 'all'
    overlap_threshold: float | None = None
    seed: int = 42
    operator: str = 'slerp'
    alpha: float | None = .1

    def __post_init__(self):
        validate_operator(self.operator, self.rho if self.operator == 'additive' else self.alpha)
        if self.direction not in DIRECTION_ROLES:
            raise ValueError('direction must be T-S or T-R')
        if self.granularity not in ('global', 'per_question') or self.extractor != 'initial':
            raise ValueError('This screen uses global/per_question directions from frozen S0')
        if (not self.layers or any(type(layer) is not int or layer < 0 for layer in self.layers)
                or len(set(self.layers)) != len(self.layers)):
            raise ValueError('Layers must be unique nonnegative block indices')
        if not math.isfinite(self.rho) or self.rho < 0 or self.seed != 42:
            raise ValueError('Use a nonnegative rho and seed42')
        if self.gate not in ('all', 'static_overlap'):
            raise ValueError('Only all/static_overlap gates are supported')
        if self.gate == 'static_overlap' and (self.overlap_threshold is None or not 0 <= self.overlap_threshold <= 1):
            raise ValueError('Static overlap needs a frozen threshold in [0,1]')

    @property
    def roles(self):
        return DIRECTION_ROLES[self.direction]

    def to_dict(self):
        return asdict(self)
