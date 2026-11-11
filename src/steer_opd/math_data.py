import json
from pathlib import Path


TASK_SUFFIX = " Please reason step by step, and put your final answer within \\boxed{}."


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def prompt_ids(tokenizer, question, enable_thinking=None):
    kwargs = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": question + TASK_SUFFIX}],
        tokenize=True, return_dict=False, add_generation_prompt=True,
        **kwargs,
    )


_RETHINKING_GRADER = None


def grade_rethinking_math(answer, text):
    """Reuse the official standalone evaluator's rule-based answer checker."""
    global _RETHINKING_GRADER
    if _RETHINKING_GRADER is None:
        import importlib.util

        path = (Path(__file__).resolve().parents[2] /
                "ref/ref_rep/Rethinking-OPD/scripts/val/eval/utils.py")
        spec = importlib.util.spec_from_file_location("_rethinking_answer_checker", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _RETHINKING_GRADER = module.grade_answer_verl
    return {"correct": bool(_RETHINKING_GRADER(text, str(answer))),
            "grading_status": "rethinking_rule_graded"}


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
