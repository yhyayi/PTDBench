"""Dense GSM8K reward for this run.

verl's stock gsm8k scorer is strict: it only accepts a literal ``#### <number>``
in the last 300 characters, so an untuned Qwen3-1.7B scores around 3% and most
GRPO groups come back all-zero, which carries no gradient at all.

This splits the signal in two so every group has something to learn from:
  0.2  for emitting the requested ``#### <number>`` marker at all
  0.8  for the final number being right (read leniently: last number wins)

``acc`` is therefore a format-independent correctness rate and ``format`` is the
compliance rate; ``score`` is the weighted sum the run optimises.
"""

import re

_STRICT = re.compile(r"#### (\-?[0-9\.\,]+)")
_NUMBER = re.compile(r"(-?[0-9][0-9,]*\.?[0-9]*)")


def _normalise(text: str) -> str:
    return text.replace(",", "").replace("$", "").rstrip(".")


def _last_number(text: str):
    matches = _NUMBER.findall(text)
    return _normalise(matches[-1]) if matches else None


def compute_score(data_source=None, solution_str=None, ground_truth=None, extra_info=None, **kwargs):
    solution_str = solution_str or ""
    truth = _normalise(str(ground_truth or ""))

    score = 0.0
    strict = _STRICT.findall(solution_str[-300:])
    if strict:
        score += 0.2
        answer = _normalise(strict[-1])
    else:
        answer = _last_number(solution_str)

    correct = answer is not None and answer == truth
    if correct:
        score += 0.8

    return {"score": score, "acc": float(correct), "format": float(bool(strict))}
