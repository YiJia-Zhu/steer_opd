from steer_opd.math_data import grade_math


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    result = grade_math(ground_truth, solution_str)
    value = float(result["correct"])
    return {"score": value, "acc": value}
