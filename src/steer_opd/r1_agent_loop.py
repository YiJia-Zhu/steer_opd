"""Use R1's native template without inferring ChatML from its Qwen2 backbone."""

from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from verl.utils.tokenizer.continuous_token import ContinuousTokenBuilder


class R1SingleTurnAgentLoop(SingleTurnAgentLoop):
    def __init__(self, *args, hf_model_type=None, **kwargs):
        if hf_model_type != "qwen2":
            raise ValueError("This loop is scoped to the inspected R1-Distill-Qwen checkpoint")
        # None selects the generic builder, which renders the tokenizer's own
        # template and appends the exact sampled token IDs. The HF model config
        # remains qwen2; we only bypass the architecture-to-ChatML inference.
        super().__init__(*args, hf_model_type=None, **kwargs)
        if type(self.continuous_token_builder) is not ContinuousTokenBuilder:
            raise RuntimeError("Expected the native-template generic builder")
        if self.tokenizer.convert_tokens_to_ids("<|im_end|>") is not None:
            raise ValueError("Unexpected ChatML tokenizer; recheck this loop's scope")
