"""Known-mask EasySteer authoring and fixed-prefix scoring adapters.

No online teacher, probe or changing selector. Import the optional engine
only when a caller actually selects it, and fail rather than silently falling
back to a different backend. False explicitly disables inherited steering.
"""

import os
from pathlib import Path
import sys

import torch

from steer_opd.easysteer_plugin import ALGORITHM, register
from steer_opd.operators import validate_operator
from steer_opd.static_masks import selector_for_predictions


def require_easysteer():
    if not register():
        raise RuntimeError("This environment lacks the supported EasySteer model_hooks API; upstream vLLM0.32 is not EasySteer. Install the fork and the project general plugin in an isolated environment.")


def require_easysteer_gpu():
    """Keep GPU admission, without requiring the separate OPD vLLM0.32 pin."""
    require_easysteer()
    if not os.environ.get("STEER_OPD_GPU_ADMISSION"):
        raise RuntimeError("Fresh idle-GPU admission is required for EasySteer")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("The static EasySteer backend requires exactly one admitted GPU")


def build_spec(row, gates, vectors, *, operator="slerp", alpha=0.1, mode="fixed_prefix"):
    validate_operator(operator, alpha if operator == "slerp" else 1.0)
    clause = selector_for_predictions(row, gates, mode=mode)
    if clause is None or (operator == "slerp" and alpha == 0):
        return False
    vectors = {layer: value for layer, value in vectors.items() if bool(torch.count_nonzero(value))}
    if not vectors:
        return False
    require_easysteer()
    from vllm.model_hooks.steering.api import ApplySpec, SteeringSpec, VectorSpec
    from vllm.model_hooks.steering.payloads import DirectionVector
    return SteeringSpec(vectors=[VectorSpec(
        data=DirectionVector(vectors), layers=list(vectors),
        algorithm=ALGORITHM if operator == "slerp" else "direct",
        scale=alpha if operator == "slerp" else 1.0, normalize=False,
        apply=ApplySpec(**clause), name=f"static-{row.get('key', row['id'])}")])


def engine_options():
    require_easysteer()
    site = Path(__file__).resolve().parents[2] / ".runtime/easysteer-plugin"
    if not list(site.glob("steer_opd_easysteer_plugin-*.dist-info/entry_points.txt")):
        raise RuntimeError("Install the project worker plugin metadata into .runtime/easysteer-plugin before launching EasySteer")
    # Spawned engine/worker processes discover the same general plugin before
    # validating their steering configuration. No shared environment is edited.
    if str(site) not in sys.path:
        sys.path.insert(0, str(site))
    existing = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    if str(site) not in existing:
        os.environ["PYTHONPATH"] = os.pathsep.join([str(site), *filter(None, existing)])
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    return {"steer_algorithms": ["direct", ALGORITHM], "steer_graph_mode": "split",
            "enable_prefix_caching": False}


def fixed_prefix_inputs(rows):
    return [{"prompt_token_ids": row["prompt_token_ids"]+row["token_ids"][:-1]} for row in rows]


def fixed_prefix_scores(llm, rows, specs, *, k=16):
    """Batch all source prefixes; one output step supplies the last logits.

    The sampled output is discarded. Prompt logprobs give predictions 0..N-2;
    the single output distribution gives N-1, including a source EOS target.
    This is fixed-prefix scoring, not a second controlled answer rollout.
    """
    if len(rows) != len(specs) or not rows:
        raise ValueError("Need one explicit steering choice per source prefix")
    config = llm.llm_engine.vllm_config
    if config.model_config.logprobs_mode != "raw_logprobs":
        raise ValueError("Static scoring requires raw, unwarped model logprobs")
    if config.cache_config.enable_prefix_caching:
        raise ValueError("Disable prefix caching until per-request steering cache identity is accepted")
    from vllm import SamplingParams
    params = SamplingParams(max_tokens=1, temperature=0, prompt_logprobs=k, logprobs=k,
                            ignore_eos=True, detokenize=False)
    outputs = llm.generate(fixed_prefix_inputs(rows), params, steering=specs, use_tqdm=False)
    if len(outputs) != len(rows):
        raise RuntimeError("Engine returned an incomplete fixed-prefix batch")
    results = []
    for row, output in zip(rows, outputs):
        expected = row["prompt_token_ids"]+row["token_ids"][:-1]
        if list(output.prompt_token_ids) != expected:
            raise ValueError("Engine changed fixed-prefix tokens or output order")
        n, p = len(row["token_ids"]), len(row["prompt_token_ids"])
        if output.prompt_logprobs is None or len(output.prompt_logprobs) != len(expected):
            raise ValueError("Missing fixed-prefix prompt logprobs")
        steps = list(output.prompt_logprobs[p:p+n-1])
        if len(output.outputs) != 1 or not output.outputs[0].logprobs or len(output.outputs[0].logprobs) != 1:
            raise ValueError("Missing final prediction distribution")
        steps.append(output.outputs[0].logprobs[0])
        ids, logps = [], []
        for entries in steps:
            if entries is None:
                raise ValueError("Incomplete source prediction distribution")
            ranked = sorted(((value.rank, token, float(value.logprob)) for token, value in entries.items()
                             if value.rank is not None and value.rank <= k), key=lambda x: x[0])
            if [rank for rank, _, _ in ranked] != list(range(1, k+1)):
                raise ValueError("Missing or ambiguous raw Top-K ranks")
            ids.append([token for _, token, _ in ranked])
            logps.append([lp for _, _, lp in ranked])
        results.append({"ids": torch.tensor(ids, dtype=torch.int32),
                        "logps": torch.tensor(logps, dtype=torch.float32),
                        "backend": "easysteer_static_fixed_prefix", "current_step_probe": False,
                        "parameter_updates": 0})
    return results
