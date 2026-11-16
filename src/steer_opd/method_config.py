"""Reader-space T-S/student and S-T/teacher intervention definitions."""
from dataclasses import asdict, dataclass
import math

from steer_opd.operators import validate_operator


DIRECTION_VERSION = 'sv_training_teacher_self_answer_mean_v2'
DIRECTION_ROLES = {'T-S': ('teacher', 'student'), 'S-T': ('student', 'teacher')}
TEACHER_DIRECTION_VERSION = 'sv_training_student_teacher_answer_mean_teacher_space_v1'
DIRECTION_VERSIONS = {'T-S': DIRECTION_VERSION, 'S-T': TEACHER_DIRECTION_VERSION}


def reader_direction(role):
    if role not in ('student', 'teacher'):
        raise ValueError('The direction reader must be student or teacher')
    return 'T-S' if role == 'student' else 'S-T'


def reader_layers(config, role):
    return config['layers'] if role == 'student' else config['teacher_layers']


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
            raise ValueError('The current screen supports only T-S or S-T')
        if self.granularity not in ('global', 'per_question') or self.extractor != 'initial':
            raise ValueError('Use global/per_question directions from the frozen reader')
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
