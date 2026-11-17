"""Keep every single-turn student sampling request at the user's seed 42."""
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop


class ConstantSeedAgentLoop(SingleTurnAgentLoop):
    async def run(self,sampling_params,priority=0,**kwargs):
        params={**sampling_params,'seed':42}
        return await super().run(params,priority=priority,**kwargs)
