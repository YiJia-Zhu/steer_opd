"""VERL DataProto bridge and public set_loss_fn hook; no CUDA imports on preview."""
import torch

from steer_opd.training.config import TrainingLossConfig
from steer_opd.training.losses import distillation_loss


def attach_advantages(batch, scored, settings=None):
    """Bind frozen scorer outputs to exact generated IDs, then mask native positions.

    old_log_probs are recomputed by VERL for this batch before any update.
    Scoring replies contain the exact source IDs; order/shape mismatches fail.
    """
    settings = settings or TrainingLossConfig()
    tensors = batch.batch
    response = tensors['responses']
    original_mask = tensors['response_mask'].bool()
    lengths = tensors['attention_mask'][:, -response.shape[1]:].sum(-1).tolist()
    if len(scored) != len(response):raise ValueError('Scorer reply changed batch size')
    advantage = torch.zeros_like(tensors['old_log_probs'])
    effective = torch.zeros_like(original_mask)
    counts = []
    for i, (reply, length) in enumerate(zip(scored, lengths)):
        ids = response[i, :length].tolist()
        if reply['alignment']['source_token_ids'] != ids:
            raise ValueError('Scorer response is not bound to actual student rollout IDs')
        if not any(reply['alignment']['alignment_mask']):
            counts.append(0)
            continue
        result = distillation_loss(reply['alignment'], tensors['old_log_probs'][i, :length],
            reply['teacher_logps'], settings=settings, reference_logps=reply.get('reference_logps'),
            student_anchor_logps=reply.get('student_anchor_logps'), response_mask=original_mask[i, :length])
        advantage[i, :length] = result.advantages.to(advantage)
        effective[i, :length] = result.alignment_mask.to(effective.device)
        counts.append(result.valid_tokens)
    if not sum(counts):raise ValueError('Entire rollout batch has no aligned training tokens')
    tensors['advantages'] = advantage.detach()
    tensors['returns'] = advantage.detach().clone()
    tensors['token_level_scores'] = advantage.detach().clone()
    tensors['token_level_rewards'] = advantage.detach().clone()
    tensors['response_mask'] = effective.to(tensors['response_mask'].dtype)
    batch.meta_info['steer_opd_valid_tokens'] = sum(counts)
    return batch


def sampled_policy_loss(log_prob, advantages, response_mask, *, global_valid_tokens, dp_size=1):
    """Scale summed microbatch losses so FSDP's DP mean is a global token mean."""
    if log_prob.shape != advantages.shape or log_prob.shape != response_mask.shape:
        raise ValueError('Policy tensors must have identical native-student shapes')
    if global_valid_tokens <= 0 or dp_size < 1:raise ValueError('Invalid global loss normalization')
    mask = response_mask.bool()
    if not torch.isfinite(log_prob[mask]).all() or not torch.isfinite(advantages[mask]).all():
        raise ValueError('Nonfinite valid student policy loss inputs')
    return -(log_prob[mask].float() * advantages.detach()[mask].float()).sum() * dp_size / global_valid_tokens


def verl_loss(model_output, data, dp_group=None):
    """Pinned ActorRolloutRefWorker.set_loss_fn interface, including packed outputs."""
    from verl.workers.utils.padding import no_padding_2_padding
    from verl.utils.metric import Metric, AggregationType
    log_prob = no_padding_2_padding(model_output['log_probs'], data)
    fields = data.select('advantages', 'response_mask').to_padded_tensor()
    loss = sampled_policy_loss(log_prob, fields['advantages'], fields['response_mask'],
        global_valid_tokens=data['batch_num_tokens'], dp_size=data['dp_size'])
    return loss, {'actor/pg_loss': Metric(value=loss.detach(), aggregation=AggregationType.SUM)}
