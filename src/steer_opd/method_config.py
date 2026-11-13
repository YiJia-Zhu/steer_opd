"""Independent, validated steering axes; legacy first-run defaults stay fixed."""

from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_HALF_UP
import json
import math
from pathlib import Path


DIRECTION_ROLES = {"T-S": ("teacher", "student"), "T-R": ("teacher", "reference"),
                   "R-S": ("reference", "student")}


def resolve_layer_depths(num_hidden_layers, depths):
    """Relative output-block depth -> zero-based index, nearest with ties up.

    A depth p denotes the round(p*L)-th block (counted from one). Refuse
    collisions instead of silently turning a multi-layer trial into fewer layers.
    """
    if isinstance(num_hidden_layers, bool) or not isinstance(num_hidden_layers, int) or num_hidden_layers < 1:
        raise ValueError("Student block count must be a positive integer")
    if not depths:
        raise ValueError("Relative layer depths must not be empty")
    layers = []
    for depth in depths:
        if (isinstance(depth, bool) or not isinstance(depth, (int, float))
                or not math.isfinite(depth) or not 0 < depth <= 1):
            raise ValueError("Layer depths must be fractions in (0, 1], e.g. 0.5 for 50%")
        # Decimal text keeps configured half ties (e.g. .58*25=14.5) from
        # rounding downward due to binary floating-point representation.
        block = int((Decimal(str(depth))*num_hidden_layers).to_integral_value(rounding=ROUND_HALF_UP))
        layers.append(max(0, min(num_hidden_layers-1, block-1)))
    if len(set(layers)) != len(layers):
        raise ValueError("Relative depths collapse to the same block; choose distinct depths for this student")
    return tuple(layers)


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
    operator: str = "additive"
    alpha: float | None = None

    def __post_init__(self):
        from steer_opd.operators import validate_operator
        strength = self.rho if self.operator == "additive" else self.alpha if self.alpha is not None else float("nan")
        validate_operator(self.operator, strength)
        if self.operator == "additive" and self.alpha is not None:
            raise ValueError("Use rho for additive steering and alpha for Slerp")
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
        if self.gate not in ("all", "overlap", "static_overlap", "entropy", "random", "late", "matched_random", "matched_position_random"):
            raise ValueError("Unknown gate")
        if self.gate in ("overlap", "static_overlap", "matched_random", "matched_position_random") and (self.overlap_threshold is None or not 0 <= self.overlap_threshold <= 1):
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
        result = asdict(self)
        if self.operator == "additive":
            # Preserve existing additive bank serialization and identities.
            result.pop("operator")
            result.pop("alpha")
        return result


def resolve_method(cfg):
    if cfg.get("injection", "all_response_prediction_states_rollout_and_replay") not in (
        "all_response_prediction_states_rollout_and_replay", "recorded_causal_gate_mask_rollout_and_replay",
    ):
        raise ValueError("Requested injection phase/schedule is not implemented by this campaign")
    options = dict(cfg.get("steering_method", {}))
    depths = options.pop("layer_depths", None)
    if depths is not None:
        if not cfg.get("student"):
            raise ValueError("Relative layer depths require the student model config")
        model = Path(cfg["student"])
        if not model.is_absolute():
            model = Path(__file__).resolve().parents[2] / model
        model_config = json.loads((model / "config.json").read_text())
        text_config = model_config.get("text_config", model_config)
        layers = resolve_layer_depths(text_config.get("num_hidden_layers"), depths)
        if "layers" in options and tuple(options["layers"]) != layers:
            raise ValueError("Explicit layers do not match the student's relative-depth mapping")
        options["layers"] = layers
    options.setdefault("direction", cfg.get("direction", "T-S"))
    options.setdefault("layers", [cfg.get("layer", 13)])
    options.setdefault("rho", cfg.get("rho", 0.03))
    options.setdefault("seed", cfg.get("seed", 42))
    options["layers"] = tuple(options["layers"])
    method = Method(**options)
    if (method.operator != "additive" or method.gate == "static_overlap") and cfg.get("stage") != "static_overlap_screen":
        raise ValueError("Slerp/static masks require the explicit static_overlap_screen entrypoint; legacy campaigns are unchanged")
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
