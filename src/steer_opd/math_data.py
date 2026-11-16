"""Frozen mathematical-question chat template; no data download or evaluation."""
TASK_SUFFIX = " Please reason step by step, and put your final answer within \\boxed{}."


def prompt_text(tokenizer, question, enable_thinking=None):
    kwargs = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": question + TASK_SUFFIX}],
        tokenize=False, add_generation_prompt=True, **kwargs)


def prompt_ids(tokenizer, question, enable_thinking=None):
    kwargs = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": question + TASK_SUFFIX}],
        tokenize=True, return_dict=False, add_generation_prompt=True,
        **kwargs,
    )
