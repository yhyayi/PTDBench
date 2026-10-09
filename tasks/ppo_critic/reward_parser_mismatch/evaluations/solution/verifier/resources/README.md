# Metric-only solution verifier

`test.sh` extracts task metrics only from fresh `$PTDBENCH_RESULTS/experiment_logs`, including offline W&B files under `$PTDBENCH_RESULTS/experiment_logs/wandb`. It does not grade `submission.json`, code rules, rubrics, or an LLM judgment.

It reads every log under `experiment_logs` and assigns a weighted score:

- validation: the **terminal** `val-core/math__math/acc/mean@1` — the value at the latest step `<= final_step` — must be at least `0.10` (10%) to score `1`, otherwise `0`. The terminal value is used, **not** a peak, so a run that briefly rises and then collapses does not pass. (For this bug the metric is `0.0` at every step, so terminal and peak coincide at `0.0`.)
- training reward: the mean `critic/rewards/mean` over training steps must be strictly above `0.05` to score `1`, otherwise `0`.
- final `score = 0.8 * validation + 0.2 * training reward`.

Pass requires both components (`score == 1.0`). The framework still records the weighted float when only one component passes.

The verifier writes `experiment_metrics.json`, `metrics.json`, and `result.json`. `test.sh` runs `tests/grade_submission.py`, which reads `resources/grader_spec.json`.

This grader is adapted from `verl_implementation/torch_functional` with two deviations required by the VIMPO PPO-Critic log format: it tolerates numpy scalar reprs (`...:np.float64(0.0)`) and captures every val-core metric on each `step:N` line (the template captured only the first), and it selects the terminal value via `select_final` rather than the peak.

> **Calibration note (maintainers).** `historical_final_baseline` is the buggy run's real terminal value (`0.0`). `required_final_score` and `required_training_reward` are evidence-based conservative drafts and are **not yet calibrated** against a real oracle-fixed rerun. Calibrate before using this task for scoring; `../oracle/task.yaml` carries `needs_manual_review: true`.
