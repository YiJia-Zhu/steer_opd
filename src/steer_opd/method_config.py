"""Independent, validated steering axes; legacy first-run defaults stay fixed."""

from dataclasses import asdict, dataclass
import math


DIRECTION_ROLES = {"T-S": ("teacher", "student"), "T-R": ("teacher", "reference"),
                   "R-S": ("reference", "student")}


@dataclass(frozen=True)
class Method:
    direction: str = "T-S"
    granularity: str = "global"
    extractor: str = "initial"
    refresh_interval: int = 1
    layers: tuple = (13,)
    rho: float = 0.03
    gate: str = "all"
    overlap_threshold: float | None = None
    entropy_threshold: float | None = None
    random_probability: float | None = None
    start_token: int | None = None
    matching_bin_width: int = 256
    vector_assignment: str = "own"
    direction_control: str = "positive"
    seed: int = 42

    def __post_init__(self):
        if self.direction not in DIRECTION_ROLES:
            raise ValueError("direction must be T-S, T-R or R-S")
        if self.granularity not in ("global", "per_question") or self.extractor not in ("initial", "current"):
            raise ValueError("Unknown direction granularity/extractor")
        if self.refresh_interval < 1 or (self.extractor == "initial" and self.refresh_interval != 1):
            raise ValueError("Refresh interval applies to current-student extraction only")
        if not self.layers or len(set(self.layers)) != len(self.layers) or min(self.layers) < 0:
            raise ValueError("Layers must be unique nonnegative block indices")
        if not math.isfinite(self.rho) or self.rho < 0:
            raise ValueError("rho must be finite and nonnegative")
        if self.gate not in ("all", "overlap", "entropy", "random", "late", "matched_random", "matched_position_random"):
            raise ValueError("Unknown gate")
        if self.gate in ("overlap", "matched_random", "matched_position_random") and (self.overlap_threshold is None or not 0 <= self.overlap_threshold <= 1):
            raise ValueError("Overlap gate needs a frozen training/dev threshold in [0,1]")
        if self.gate == "entropy" and (self.entropy_threshold is None or not math.isfinite(self.entropy_threshold)
                                       or self.entropy_threshold < 0):
            raise ValueError("Entropy gate needs a frozen nonnegative signed-entropy threshold")
        if self.gate in ("random", "matched_random", "matched_position_random") and (self.random_probability is None or not 0 <= self.random_probability <= 1):
            raise ValueError("Random gate needs a causally calibrated probability")
        if self.gate == "late" and (self.start_token is None or self.start_token < 0):
            raise ValueError("Late gate needs an advance-fixed token boundary")
        if self.vector_assignment not in ("own", "other_question"):
            raise ValueError("Unknown question/vector assignment")
        if self.vector_assignment != "own" and self.granularity != "per_question":
            raise ValueError("Other-question control requires per-question vectors")
        if self.direction_control not in ("positive", "negative", "random"):
            raise ValueError("Unknown direction control")
        if self.matching_bin_width < 1:
            raise ValueError("Matching bins need a positive absolute width")

    @property
    def roles(self):
        return DIRECTION_ROLES[self.direction]

    @property
    def online_gate(self):
        return self.gate in ("overlap", "entropy", "matched_random", "matched_position_random")

    def extraction_step(self, learning_step):
        if self.extractor == "initial":
            return 0
        return ((learning_step - 1) // self.refresh_interval) * self.refresh_interval

    def to_dict(self):
        return asdict(self)


def resolve_method(cfg):
    if cfg.get("injection", "all_response_prediction_states_rollout_and_replay") not in (
        "all_response_prediction_states_rollout_and_replay", "recorded_causal_gate_mask_rollout_and_replay",
    ):
        raise ValueError("Requested injection phase/schedule is not implemented by this campaign")
    options = dict(cfg.get("steering_method", {}))
    options.setdefault("direction", cfg.get("direction", "T-S"))
    options.setdefault("layers", [cfg.get("layer", 13)])
    options.setdefault("rho", cfg.get("rho", 0.03))
    options.setdefault("seed", cfg.get("seed", 42))
    options["layers"] = tuple(options["layers"])
    method = Method(**options)
    if "reference" in method.roles and not cfg.get("reference"):
        raise ValueError("T-R/R-S needs an explicit frozen reference model")
    return method


def intervention_phases(cfg, group, step):
    phase = cfg.get("steering_phase", "both")
    if phase not in ("both", "rollout_only", "learning_only"):
        raise ValueError("Unknown steering_phase")
    if phase != "both" and cfg.get("loss_objective") != "candidate_rkl":
        raise ValueError("Different behavior/learning steering requires the explicit candidate_rkl state-replay objective")
    cutoff = cfg.get("assistance_until_step", cfg.get("steps", 50))
    enabled = group == "steer" and step <= cutoff
    return enabled and phase != "learning_only", enabled and phase != "rollout_only"


def uses_legacy_direction(cfg):
    method = resolve_method(cfg)
    return ("steering_method" not in cfg and method.direction == "T-S"
            and method.granularity == "global" and method.extractor == "initial" and method.gate == "all")
