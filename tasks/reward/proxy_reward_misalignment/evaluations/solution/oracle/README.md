# Canonical solution oracle

## Root cause

- `ptdbench_case/type3_length_reward_v2_fn.py` (`compute_score`)
  - kind: `symbol`
  - observed: for the controlled training source the training `score` is a
    brevity proxy, `proxy_reward = max(0.0, min(1.0, 1.0 - n_chars/4000.0))`.
  - expected: score the training source by mathematical correctness.
  - note: the reward is wired into training by `run_buggy.sh` via
    `reward.custom_reward_function.path`/`name=compute_score`. Validation uses
    genuine correctness, so training and the target task are misaligned.

## Observable impact

- Primary source evaluation metric: `val-core/math__math/acc/mean@1`.
- `critic/rewards/mean` (the proxy) rises (~0.52 → 0.67) while held-out accuracy
  declines (0.78 → 0.59) and `response_length/mean` shrinks. PPO runs all 15
  steps.

## Trigger conditions

- `run_buggy.sh` sets
  `reward.custom_reward_function.path=ptdbench_case/type3_length_reward_v2_fn.py`;
  its training-source branch rewards brevity, not correctness.

## Required remediation

- Make the controlled training source score mathematical correctness, not
  response length. A short incorrect answer must score 0; both short and long
  correct boxed answers must score 1. Keep the
  `compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs)`
  signature. The reference `fixed_reward_fn.py` routes the training source
  (`ptdbench_type3_proxy_train_v2`) through `default_compute_score` on
  `math__math`, so training and validation share one correctness scorer. Do not
  alter the critic/actor LR, validation data, or the preserved evidence.

Machine grading reads only the new experiment logs. Scoring is weighted:

- validation: the **terminal** (final-step) `val-core/math__math/acc/mean@1`
  must be at least `required_final_score` (see `grader_spec.json`). The terminal
  value is used, **not** a peak, because the buggy run peaks at step 0
  (`0.78125`) before the proxy corrupts the policy; its terminal value is the
  historical baseline `0.59375`.
- training reward: mean `critic/rewards/mean` over training steps must be
  strictly above `required_training_reward`. For this task the proxy makes this
  component non-discriminating; the terminal validation accuracy is the real
  discriminator.
- final `score = 0.8 * validation + 0.2 * training_reward`.

Pass requires both components (`score == 1.0`). The framework still records the
weighted float when only one component passes.

> **Threshold calibration note.** `required_final_score` (`0.65`) is an
> evidence-based conservative draft placed just above the hacked terminal
> (`0.59375`) and below the step-0 baseline (`0.78125`). It is **not** verified
> against a real correctness-fixed rerun and may need adjustment. See
> `task.yaml` (`needs_manual_review: true`).
