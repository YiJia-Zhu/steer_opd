"""Retain sampler Top16 during the existing student generation call."""
import torch

from steer_opd.training.verl_agent import TrajectorySeedAgentLoop


class Top16AgentLoop(TrajectorySeedAgentLoop):
    async def run(self,sampling_params,priority=0,**kwargs):
        monitoring=not str(kwargs.get('data_source','')).startswith('steer_eval_')
        params={**sampling_params}
        if monitoring:params.update(topk_log_probs=16,logprobs=True)
        result=await super().run(params,priority=priority,**kwargs)
        if monitoring:
            ids=result.extra_fields.get('response_topk_ids')
            logps=result.extra_fields.get('response_topk_log_probs')
            count=len(result.response_ids)
            if ids is None or logps is None or tuple(ids.shape)!=(count,16) or tuple(logps.shape)!=(count,16):
                raise ValueError('Student sampler did not return exact native Top16 for all response positions')
            result.extra_fields['response_topk_ids']=torch.as_tensor(ids,dtype=torch.int32).numpy()
            result.extra_fields['response_topk_log_probs']=torch.as_tensor(logps,dtype=torch.float32).numpy()
        return result
