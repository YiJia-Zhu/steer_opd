"""Use the trajectory seed assigned before worker dispatch; evaluation keeps 42."""
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from steer_opd.training.rollout_seeds import SEED_FIELD
from steer_opd.generation_policy import PRESENCE_FIELD, presence_penalty


class TrajectorySeedAgentLoop(SingleTurnAgentLoop):
    async def run(self,sampling_params,priority=0,**kwargs):
        if SEED_FIELD not in kwargs:raise ValueError('Missing driver-assigned rollout trajectory seed')
        if PRESENCE_FIELD not in kwargs:raise ValueError('Missing driver-assigned presence penalty')
        penalty=presence_penalty({'presence_penalty':kwargs[PRESENCE_FIELD]})
        seed=int(kwargs[SEED_FIELD]);params={**sampling_params,'seed':seed,'presence_penalty':penalty}
        result=await super().run(params,priority=priority,**kwargs)
        result.extra_fields[SEED_FIELD]=seed
        result.extra_fields[PRESENCE_FIELD]=penalty
        return result
