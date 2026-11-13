"""Install the project hook BEFORE vLLM captures decode CUDA graphs."""

import os
from types import MethodType
from pathlib import Path

import torch

from vllm.v1.worker.gpu_worker import Worker
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
from vllm.model_executor.layers.activation import SiluAndMul

from steer_opd.vllm_steering import install_worker_steering


def hf_reference_rms_norm(self, hidden, residual=None):
    """Match HF's BF16 residual sum and pre-weight rounding exactly.

    vLLM's fused norm can combine these operations in FP32. That is a valid
    inference optimization, but causes measurable rollout/training drift here.
    CUDA graphs amortize the extra native operations without changing rounding.
    """
    has_residual = residual is not None
    if has_residual:
        hidden = hidden + residual
        residual = hidden
    dtype = hidden.dtype
    value = hidden.float()
    value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.variance_epsilon)
    result = self.weight * value.to(dtype)
    return (result, residual) if has_residual else result


class SteeringWorker(Worker):
    def load_model(self, *, load_dummy_weights=False):
        super().load_model(load_dummy_weights=load_dummy_weights)
        support = int(os.environ["STEER_OPD_VLLM_SUPPORT"])
        # Crop output logits, not input embeddings: teacher-native prompt
        # control tokens remain valid input IDs. Softmax renormalizes support.
        self.model_runner.get_model().logits_processor.org_vocab_size = support
        if os.environ.get("STEER_OPD_REFERENCE_OPS") == "1":
            for module in self.model_runner.get_model().modules():
                if isinstance(module, RMSNorm):
                    module.forward = MethodType(hf_reference_rms_norm, module)
                elif isinstance(module, (RotaryEmbedding, SiluAndMul)):
                    module.forward = module.forward_native
        path = os.environ.get("STEER_OPD_VLLM_VECTOR")
        if path:
            install_worker_steering(self, path)
        capture = os.environ.get("STEER_OPD_CAPTURE_DIR")
        if capture:
            root = Path(capture)
            root.mkdir(parents=True, exist_ok=True)
            runner = self.model_runner
            self._project_capture_ids = {}
            audit_tokens = int(os.environ.get("STEER_OPD_REPLAY_AUDIT_TOKENS", "0"))
            capture_positions = set(range(audit_tokens)) if audit_tokens else {0, 1, 7, 15, 31}

            def capture_logits(module, args, logits):
                for i, request_id in enumerate(runner.input_batch.req_ids):
                    self._project_capture_ids[request_id.rsplit("-", 1)[0]] = request_id
                    position = int(runner.optimistic_seq_lens_cpu[i].item() - runner.input_batch.num_prompt_tokens[i])
                    if position in capture_positions:
                        target = root / f"{request_id}_{position}.pt"
                        temp = target.with_suffix(".tmp")
                        torch.save(logits[i].detach().float().cpu(), temp)
                        os.replace(temp, target)

            self.model_runner.get_model().logits_processor.register_forward_hook(capture_logits)

    def project_steering_status(self):
        model = self.model_runner.get_model()
        return {"installed": hasattr(model, "_project_steering_handle"),
                "nonzero": getattr(model, "_project_steering_active", False)}

    def project_capture_ids(self):
        return getattr(self, "_project_capture_ids", {})
