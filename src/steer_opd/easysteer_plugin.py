"""EasySteer general plugin: register the project's Slerp tensor algorithm.

The fork's registry and authoring capabilities both need the same name.
Use split mode: this plugin does not claim a full-graph Slerp kernel.
"""

from steer_opd.operators import slerp, validate_operator


ALGORITHM = "steer_opd_slerp"


def register():
    """Called by vLLM's general-plugin loader in parent and worker processes."""
    try:
        from vllm.model_hooks.steering.algorithms.base import BaseSteerVectorAlgorithm
        from vllm.model_hooks.steering.algorithms.registry import ALGORITHM_REGISTRY
        from vllm.model_hooks.steering.capabilities import ALGORITHM_CAPABILITIES, AlgorithmCapabilities
    except ImportError:
        return False  # Upstream vLLM and the old EasySteer API are not this fork.
    if ALGORITHM in ALGORITHM_REGISTRY:
        if not getattr(ALGORITHM_REGISTRY[ALGORITHM], "_steer_opd_slerp", False):
            raise RuntimeError("A different plugin already owns the project Slerp algorithm name")
        return True

    class SlerpAlgorithm(BaseSteerVectorAlgorithm):
        graph_family = None
        _steer_opd_slerp = True

        def set_payload(self, payload, scale_factor=1.0):
            validate_operator("slerp", float(scale_factor))
            if payload is None:
                raise ValueError("Slerp needs a direction payload")
            # Scaling v would disappear during unit normalization. Scale is
            # the interpolation angle fraction, kept separately from v.
            self.payload = payload
            self.alpha = float(scale_factor)

        def _transform(self, hidden_state, params):
            return slerp(hidden_state, params, self.alpha)

    ALGORITHM_REGISTRY[ALGORITHM] = SlerpAlgorithm
    ALGORITHM_CAPABILITIES[ALGORITHM] = AlgorithmCapabilities("direction", "none")
    return True
