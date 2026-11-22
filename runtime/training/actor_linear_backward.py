"""Supply the dense gradient for vLLM's CUDA linear override in the actor."""
from math import prod

import torch


def dense_linear_backward(input, grad_output, weight, output_mask):
    rows = prod(input.shape[:-1])
    grad_rows = grad_output.reshape(rows, weight.shape[0])
    grad_input = grad_rows.mm(weight).reshape_as(input) if output_mask[0] else None
    grad_weight = grad_rows.t().mm(input.reshape(rows, weight.shape[1])) if output_mask[1] else None
    grad_bias = grad_rows.sum(dim=0) if output_mask[2] else None
    return grad_input, grad_weight, grad_bias


def register_missing_cuda_linear_backward():
    has_kernel = torch._C._dispatch_has_kernel_for_dispatch_key
    if not has_kernel('aten::linear', 'CUDA') or has_kernel('aten::linear_backward', 'CUDA'):
        return None

    # A fused CUDA linear forward uses LinearBackward0 instead of the native
    # composite backward. Keep the Library alive for the lifetime of the worker.
    library = torch.library.Library('aten', 'IMPL')
    library.impl('linear_backward', dense_linear_backward, 'CUDA')
    return library
