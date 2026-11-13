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
    if payload.get("schema") == 2:
        return install_bank_steering(worker, payload)
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


def install_bank_steering(worker, bank):
    import numpy as np
    import torch
    from vllm.config.compilation import CUDAGraphMode
    from vllm.forward_context import get_forward_context
    from steer_opd.direction_bank import bank_vectors

    runner = worker.model_runner
    model = runner.get_model()
    if bank["method"]["gate"] != "all":
        raise ValueError("Causal gates must use the controlled reference backend")
    if bank["method"].get("operator", "additive") != "additive":
        raise ValueError("Slerp needs the explicit static engine adapter")
    per_question = bank["method"]["granularity"] == "per_question"
    question_ids = bank["valid_ids"] if per_question else [None]
    deltas = {qid: {layer: value.to(device="cuda", dtype=torch.bfloat16)
                    for layer, value in bank_vectors(bank, qid)[0].items()} for qid in question_ids}
    worker._project_question_ids = {}
    worker._project_vector_bank_ids = set(question_ids)
    handles = []
    for layer in map(int, bank["layers"]):
        active = any(bool(torch.count_nonzero(values[layer])) for values in deltas.values())
        def hook(module, args, output, layer_index=layer, enabled=active):
            if not enabled:
                return output
            hidden, residual = output
            graph = get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.FULL
            if graph:
                if per_question:
                    raise RuntimeError("Per-request vectors require eager decode in this reference implementation")
                return (hidden + residual) + deltas[None][layer_index], torch.zeros_like(residual)
            n = runner.input_batch.num_reqs
            if not n:
                return output
            starts = runner.query_start_loc.np[:n + 1]
            lengths = np.diff(starts)
            actual = int(starts[-1])
            positions = args[0]
            boundaries = np.repeat(runner.input_batch.num_prompt_tokens[:n] - 1, lengths)
            mask = torch.zeros(len(hidden), dtype=torch.bool, device=hidden.device)
            mask[:actual] = positions[:actual] >= torch.as_tensor(boundaries, device=hidden.device)
            if per_question:
                values = []
                for request_id in runner.input_batch.req_ids[:n]:
                    qid = worker._project_question_ids.get(request_id.rsplit("-", 1)[0])
                    if qid not in deltas:
                        raise ValueError("vLLM request has no bound question direction")
                    values.append(deltas[qid][layer_index])
                delta = torch.zeros_like(hidden)
                delta[:actual] = torch.repeat_interleave(torch.stack(values), torch.as_tensor(lengths, device=hidden.device), dim=0)
            else:
                delta = deltas[None][layer_index]
            combined = (hidden + residual) + delta
            return torch.where(mask[:, None], combined, hidden), torch.where(mask[:, None], 0, residual)
        handles.append(model.model.layers[layer].register_forward_hook(hook))
    model._project_steering_handle = handles[0]
    model._project_steering_handles = handles
    model._project_steering_active = True
    return {"bank_id": bank["bank_id"], "per_question": per_question,
            "layers": list(map(int, bank["layers"])), "backend": "vllm_v1_direction_bank"}
