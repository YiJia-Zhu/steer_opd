"""Known-mask EasySteer authoring and fixed-prefix scoring adapters.

No online teacher, probe or changing selector. Import the optional engine
only when a caller actually selects it, and fail rather than silently falling
back to a different backend. False explicitly disables inherited steering.
"""

import os
from pathlib import Path
import sys
from contextlib import nullcontext

import torch

from steer_opd.easysteer_plugin import ALGORITHM, register
from steer_opd.operators import validate_operator
from steer_opd.static_masks import selector_for_predictions


def require_easysteer():
    if not register():
        raise RuntimeError("This environment lacks the supported EasySteer model_hooks API; upstream vLLM0.32 is not EasySteer. Install the fork and the project general plugin in an isolated environment.")


def require_easysteer_gpu(max_gpus=1):
    """Keep GPU admission, without requiring the separate OPD vLLM0.32 pin."""
    require_easysteer()
    if not os.environ.get("STEER_OPD_GPU_ADMISSION"):
        raise RuntimeError("Fresh idle-GPU admission is required for EasySteer")
    if not 1 <= torch.cuda.device_count() <= max_gpus:
        raise RuntimeError("The static EasySteer backend requires exactly one admitted GPU unless two-GPU teacher scoring is explicitly selected")


def build_spec(row, gates, vectors, *, operator="slerp", alpha=0.1, mode="fixed_prefix"):
    validate_operator(operator, alpha if operator == "slerp" else 1.0)
    # The unsteered tail needs neither a selector nor direction materialization.
    if operator == "slerp" and alpha == 0:
        return False
    clause = selector_for_predictions(row, gates, mode=mode)
    if clause is None:
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


def engine_options(*,batch_invariant=False):
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
    # CPU tensor work and the progress thread precede engine startup. Forking
    # that process can deadlock in the V2 runner's pinned-memory torch.zeros.
    os.environ['VLLM_WORKER_MULTIPROC_METHOD']='spawn'
    os.environ['VLLM_BATCH_INVARIANT']='1' if batch_invariant else '0'
    return {"enable_steer_vector":True,"steer_algorithms": ["direct", ALGORITHM], "steer_graph_mode": "split",
            "enable_prefix_caching": False}


def create_engine(model_path, support_size, *, max_model_len,
                  tensor_parallel_size=1, gpu_memory_utilization=.75,
                  prefix_caching=False, seed=42, progress=None,batch_invariant=False):
    """Shared native inference/scoring engine; leave sequence scheduling automatic."""
    from steer_opd.experiment_runtime import compute_required, engine_startup_guard, cuda_toolkit
    compute_required()
    cuda = cuda_toolkit(sys.prefix)
    os.environ["CUDA_HOME"] = str(cuda)
    os.environ["PATH"] = os.pathsep.join([str(Path(sys.prefix)/"bin"), str(cuda/"bin"), os.environ.get("PATH", "")])
    require_easysteer_gpu(max_gpus=tensor_parallel_size)
    if torch.cuda.device_count() != tensor_parallel_size:
        raise ValueError("Engine tensor parallelism differs from admitted GPUs")
    os.environ["STEER_OPD_STATIC_SUPPORT"] = str(support_size)
    options = engine_options(batch_invariant=batch_invariant)
    options["enable_prefix_caching"] = prefix_caching
    if tensor_parallel_size>1:
        # The pinned TP startup stalled in FlashInfer autotune during GPU
        # testing. Skip that startup pass; retain normal inference kernels.
        options['enable_flashinfer_autotune']=False
    from steer_opd.resumable_core import read_json
    if 'vision_config' in read_json(Path(model_path)/'config.json'):
        # All experiment inputs are text. Skip unused image/video profiling.
        options['limit_mm_per_prompt']={'image':0,'video':0}
    from vllm import LLM
    with (progress.phase_scope('engine_initialization') if progress else nullcontext()), engine_startup_guard():
        engine = LLM(model=str(model_path), dtype="bfloat16", tensor_parallel_size=tensor_parallel_size, seed=seed,
        hf_overrides={"head_dtype": "float32"},
        max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization,
        logprobs_mode="raw_logprobs", max_logprobs=16, generation_config="vllm",
        worker_cls="steer_opd.easysteer_worker.StaticWorker", **options)
    if progress:
        from steer_opd.resumable_core import atomic_json
        config=engine.llm_engine.vllm_config
        atomic_json(progress.attempt/'engine_config.json', {
            'seed':seed,'tensor_parallel_size':tensor_parallel_size,'max_model_len':max_model_len,
            'dtype':'bfloat16','head_dtype':'float32','native_support_size':support_size,
            'cuda_home':str(cuda),'worker_multiproc_method':'spawn',
            'batch_invariant':batch_invariant,
            'prefix_caching':prefix_caching,'gpu_memory_utilization':gpu_memory_utilization,
            'resolved_max_num_seqs':config.scheduler_config.max_num_seqs,
            'resolved_max_num_batched_tokens':config.scheduler_config.max_num_batched_tokens})
    return engine


def fixed_prefix_inputs(rows, *, include_sampled=False):
    return [{"prompt_token_ids": row["prompt_token_ids"]+(row["token_ids"] if include_sampled else row["token_ids"][:-1])} for row in rows]


def fixed_prefix_scores(llm, rows, specs, *, k=16, include_sampled=False, progress=None):
    """Batch all source prefixes; one output step supplies the last logits.

    The sampled output is discarded. Prompt logprobs give predictions 0..N-2;
    the single output distribution gives N-1, including a source EOS target.
    With include_sampled=True the complete response is placed in the prompt,
    so all observed-token logps are available even outside native Top-K.
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
    from steer_opd.generation_policy import DEFAULT_PRESENCE_PENALTY
    params = SamplingParams(max_tokens=1, temperature=0, prompt_logprobs=k, logprobs=k,
                            presence_penalty=DEFAULT_PRESENCE_PENALTY,
                            ignore_eos=True, detokenize=False, seed=42)
    results = {}
    def accept(request_index, output):
        row=rows[request_index]
        expected = row["prompt_token_ids"]+(row["token_ids"] if include_sampled else row["token_ids"][:-1])
        if list(output.prompt_token_ids) != expected:
            raise ValueError("Engine changed fixed-prefix tokens or output order")
        n, p = len(row["token_ids"]), len(row["prompt_token_ids"])
        if output.prompt_logprobs is None or len(output.prompt_logprobs) != len(expected):
            raise ValueError("Missing fixed-prefix prompt logprobs")
        steps = list(output.prompt_logprobs[p:p+n] if include_sampled else output.prompt_logprobs[p:p+n-1])
        if len(output.outputs) != 1 or not output.outputs[0].logprobs or len(output.outputs[0].logprobs) != 1:
            raise ValueError("Missing final prediction distribution")
        if not include_sampled:
            steps.append(output.outputs[0].logprobs[0])
        ids, logps = [], []
        for index, entries in enumerate(steps):
            if entries is None:
                raise ValueError("Incomplete source prediction distribution")
            observed = row["token_ids"][index] if include_sampled or index < n-1 else None
            if observed is None and getattr(output.outputs[0], "token_ids", None):
                observed = output.outputs[0].token_ids[0]
            # vLLM may append the observed token outside raw Top-K, including
            # a tie at its last rank. It must not displace a native Top-K item.
            extra_observed = len(entries) == k+1
            if len(entries) not in (k, k+1) or (extra_observed and observed is None):
                raise ValueError("Ambiguous Top-K/observed-token cache")
            ranked = sorted(((value.rank, token, float(value.logprob)) for token, value in entries.items()
                             if not (extra_observed and token == observed)
                             if value.rank is not None and value.rank <= k), key=lambda x: x[0])
            if [rank for rank, _, _ in ranked] != list(range(1, k+1)):
                raise ValueError("Missing or ambiguous raw Top-K ranks")
            ids.append([token for _, token, _ in ranked])
            logps.append([lp for _, _, lp in ranked])
        record = {"ids": torch.tensor(ids, dtype=torch.int32),
                        "logps": torch.tensor(logps, dtype=torch.float32),
                        "backend": "easysteer_static_fixed_prefix", "current_step_probe": False,
                        "parameter_updates": 0}
        if include_sampled:
            if any(token not in entries for token, entries in zip(row["token_ids"], steps)):
                raise ValueError("Missing observed-token raw logprob in prompt scoring")
            record["sampled_logps"] = torch.tensor([float(entries[token].logprob) for token, entries in zip(row["token_ids"], steps)])
        results[request_index]=record
    inputs=fixed_prefix_inputs(rows, include_sampled=include_sampled)
    # One uninterrupted public engine call; parse scores only after it returns.
    outputs=llm.generate(inputs, params, steering=specs,
                         use_tqdm=progress.engine_tqdm if progress else True)
    if len(outputs)!=len(rows): raise RuntimeError("Engine returned an incomplete fixed-prefix batch")
    for index, output in enumerate(outputs): accept(index, output)
    return [results[i] for i in range(len(rows))]


def crop_native_support(model, support):
    """Exclude padded head rows for Qwen3 and nested Qwen3.5 model layouts."""
    owner = model
    while not hasattr(owner, "logits_processor") and hasattr(owner, "language_model"):
        owner = owner.language_model
    processor = getattr(owner, "logits_processor", None)
    if processor is None or not 16 <= support <= processor.org_vocab_size:
        raise ValueError("Invalid native output support or unsupported model head layout")
    processor.org_vocab_size = support
