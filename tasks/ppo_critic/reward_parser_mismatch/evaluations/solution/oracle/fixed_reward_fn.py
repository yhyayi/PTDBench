import re
from verl.utils.reward_score.naive_dapo import grade_answer

def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    text = str(solution_str or "")
    truth = str(ground_truth)
    matches = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if not matches:
        return {"score": 0.0, "acc": False, "parser_matched": False}
    extracted = matches[-1].strip()
    correct, _ = grade_answer(extracted, truth)
    return {
        "score": 1.0 if correct else 0.0,
        "acc": bool(correct),
        "parser_matched": True,
    }
