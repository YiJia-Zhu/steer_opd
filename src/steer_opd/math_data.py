import json
from pathlib import Path


TASK_SUFFIX = " Please reason step by step, and put your final answer within \\boxed{}."


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def prompt_ids(tokenizer, question):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": question + TASK_SUFFIX}],
        tokenize=True, return_dict=False, add_generation_prompt=True,
    )


def grade_math(answer, text):
    from math_verify import parse, verify

    try:
        gold = parse("\\boxed{" + str(answer) + "}")
        predicted = parse(text)
        if not gold:
            return {"correct": False, "grading_status": "gold_parse_failed"}
        if not predicted:
            return {"correct": False, "grading_status": "prediction_parse_failed"}
        return {"correct": bool(verify(gold, predicted)), "grading_status": "graded"}
    except Exception as error:
        return {"correct": False, "grading_status": "verifier_exception", "error_type": type(error).__name__}
