"""VERL rollout reward placeholder; actual per-token OPD signal is attached in trainer."""


def compute_score(data_source=None, solution_str=None, ground_truth=None, extra_info=None, **kwargs):
    if data_source in ('steer_eval_MATH-500','steer_eval_AIME2025'):
        if not ground_truth:raise ValueError('Benchmark gold answer is missing')
        from verl.utils.reward_score.math_verify import compute_score as verify_math
        acc=float(verify_math(solution_str,str(ground_truth),timeout=10.))
        return {'score':acc,'acc':acc}
    if data_source!='steer_opd':raise ValueError(f'Unknown reward data source: {data_source}')
    return 0.0
