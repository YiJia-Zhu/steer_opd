"""Project-local vLLM hook; no installed vLLM files are modified.

Use the v1 runner with eager prefill and optional decode-only CUDA graphs.
The original prompt boundary is taken from the
current input batch, so chunked/recomputed prefills retain correct masks.
"""


def install_worker_steering(worker, vector_path):
    import numpy as np
    import torch
    from vllm.config.compilation import CUDAGraphMode
    from vllm.forward_context import get_forward_context

    if getattr(worker, "use_v2_model_runner", False):
        raise RuntimeError("Steering was validated for vLLM v1 model runner only")
    runner = worker.model_runner
    model = runner.get_model()
    payload = torch.load(vector_path, map_location="cpu", weights_only=True)
    delta = payload["delta"].to(device="cuda", dtype=torch.bfloat16)
    active = bool(torch.count_nonzero(delta).item())
    layer_index = payload["layer"]
    if hasattr(model, "_project_steering_handle"):
        raise RuntimeError("Duplicate steering installation")

    def hook(module, args, output):
        if not active:
            return output  # Exact zero-strength baseline, including fused rounding.
        hidden, residual = output
        positions = args[0]
        if get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.FULL:
            if worker.vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.FULL_DECODE_ONLY:
                raise RuntimeError("Full prefill CUDA graphs are not supported by the steering hook")
            # Every real row in a decode-only graph predicts an answer token.
            # Padding rows never contribute outputs/KV to a real request.
            return (hidden + residual) + delta, torch.zeros_like(residual)
        n = runner.input_batch.num_reqs
        if n == 0:
            return output  # Memory-profiling dummy forward, not an actual request.
        starts = runner.query_start_loc.np[:n + 1]
        actual = int(starts[-1])
        if actual > len(hidden):
            raise RuntimeError("vLLM token metadata does not match decoder states")
        boundaries = np.repeat(runner.input_batch.num_prompt_tokens[:n] - 1, np.diff(starts))
        mask = torch.zeros(len(hidden), dtype=torch.bool, device=hidden.device)
        mask[:actual] = positions[:actual] >= torch.as_tensor(boundaries, device=hidden.device)
        # vLLM defers its final residual addition to the NEXT block. Recombine
        # before adding delta to match the HF block-output hook, then let the
        # next fused norm add zero at injected positions.
        combined = (hidden + residual) + delta
        return torch.where(mask[:, None], combined, hidden), torch.where(mask[:, None], 0, residual)

    model._project_steering_handle = model.model.layers[layer_index].register_forward_hook(hook)
    model._project_steering_active = active
    return {"layer": layer_index, "delta_norm": float(delta.float().norm()),
            "mask": "positions >= original_prompt_length - 1", "backend": "vllm_v1_fixed_direction"}


def worker_steering_status(worker):
    return {"installed": hasattr(worker.model_runner.get_model(), "_project_steering_handle")}
