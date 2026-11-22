"""Use the trajectory seed assigned before worker dispatch; evaluation keeps 42."""
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from steer_opd.training.rollout_logprobs import STOPPED_LENGTH_FIELD
from steer_opd.training.rollout_seeds import SEED_FIELD
from steer_opd.generation_policy import PRESENCE_FIELD, presence_penalty


class _RolloutServer:
    """Carry the server's stop information through the single-turn output."""

    def __init__(self, server):
        self.server = server

    async def generate(self, **kwargs):
        output = await self.server.generate(**kwargs)
        output.extra_fields[STOPPED_LENGTH_FIELD] = (
            len(output.token_ids) if output.stop_reason == 'completed' else None
        )
        return output


class TrajectorySeedAgentLoop(SingleTurnAgentLoop):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.server_manager = _RolloutServer(self.server_manager)

    async def run(self,sampling_params,priority=0,**kwargs):
        if SEED_FIELD not in kwargs:raise ValueError('Missing driver-assigned rollout trajectory seed')
        if PRESENCE_FIELD not in kwargs:raise ValueError('Missing driver-assigned presence penalty')
        penalty=presence_penalty({'presence_penalty':kwargs[PRESENCE_FIELD]})
        seed=int(kwargs[SEED_FIELD]);params={**sampling_params,'seed':seed,'presence_penalty':penalty}
        result=await super().run(params,priority=priority,**kwargs)
        result.extra_fields[SEED_FIELD]=seed
        result.extra_fields[PRESENCE_FIELD]=penalty
        return result
