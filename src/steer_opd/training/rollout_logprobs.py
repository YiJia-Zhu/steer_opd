"""Reuse raw model scores at sampled tokens, not transformed sampling scores."""

SOURCE = 'vllm_rollout_raw_logprobs_v1'
STOPPED_LENGTH_FIELD = 'steer_opd_stopped_response_length'


def bind_old_log_probs(batch):
    """Validate generated positions and bind a detached old-policy tensor.

    The response IDs and logprobs travel together in VERL's rollout batch.
    Padding may contain arbitrary values; every actual generated position,
    including EOS, must have a finite, nonpositive observed-token logprob.
    Teacher alignment later narrows the loss mask without changing these IDs.
    """
    import torch

    tensors = batch.batch
    if 'rollout_log_probs' not in tensors:
        raise ValueError('Missing vLLM rollout_log_probs; old-policy recomputation is disabled')
    values = tensors['rollout_log_probs']
    responses, mask = tensors['responses'], tensors['response_mask']
    if (not isinstance(values, torch.Tensor) or not values.is_floating_point()
            or values.ndim != 2 or values.shape != responses.shape or mask.shape != responses.shape):
        raise ValueError('Rollout logprobs must match the exact native student response shape')
    valid = values[mask.bool()]
    if not torch.isfinite(valid).all() or (valid > 1e-5).any():
        raise ValueError('Nonfinite or positive vLLM logprob at a generated response position')
    from steer_opd.generation_policy import RAW_PROBABILITY_BASIS, PRESENCE_FIELD, presence_penalty
    if batch.meta_info.get('steer_opd_probability_basis') not in (None, RAW_PROBABILITY_BASIS):
        raise ValueError('Transformed sampling logprobs cannot replace raw model logprobs')
    extras=getattr(batch,'non_tensor_batch',{})
    if PRESENCE_FIELD in extras:
        import numpy as np
        penalties=np.asarray(extras[PRESENCE_FIELD])
        if penalties.shape!=(len(responses),):
            raise ValueError('Missing or invalid per-trajectory presence penalty metadata')
        penalties=np.asarray([presence_penalty({'presence_penalty':v}) for v in penalties])
        batch.meta_info['steer_opd_raw_logprobs_are_behavior_logprobs'] = bool((penalties==0.).all())
    tensors['old_log_probs'] = values.detach()
    batch.meta_info['steer_opd_student_logprob_source'] = SOURCE
    batch.meta_info['steer_opd_probability_basis'] = RAW_PROBABILITY_BASIS
    return batch


def difference_sums(current, rollout, mask, *, global_valid_tokens, dp_size=1):
    """Detached contributions to token-weighted train-versus-rollout metrics.

    SUM over microbatches, then MEAN over DP ranks, yields the global token
    mean. Maxima instead use MAX at both aggregation levels. No probability
    cache contributes a gradient and no additional forward is performed.
    """
    import torch

    if current.shape != rollout.shape or current.shape != mask.shape:
        raise ValueError('Train/rollout logprobs and loss mask must have identical shapes')
    if global_valid_tokens <= 0 or dp_size < 1:
        raise ValueError('Invalid global logprob metric normalization')
    with torch.no_grad():
        delta = current[mask.bool()].float() - rollout[mask.bool()].float()
        if not torch.isfinite(delta).all():
            raise ValueError('Nonfinite train-versus-rollout logprob on a valid loss position')
        scale = dp_size / global_valid_tokens
        return {
            'abs_diff_mean': delta.abs().sum() * scale,
            'signed_diff_mean': delta.sum() * scale,
            'abs_diff_max': delta.abs().max() if delta.numel() else current.new_zeros(()),
        }
