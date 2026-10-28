# Metric-only solution verifier

`test.sh` extracts task metrics only from fresh `$PTDBENCH_RESULTS/experiment_logs`. It does not grade `submission.json`, code rules, rubrics, or an LLM judgment.

It reads every log under `experiment_logs` and assigns a binary score: the finite `val-core/openai/gsm8k/acc/mean@4` selected by `score_mode` over the run's evaluation points must be at least `0.746` to pass with `score == 1.0` (`score_mode: best`), otherwise `0`. `required_final_score_max` is `1.0`, which is the cap of the accuracy space and rejects fabricated values above it.

The metric is a held-out accuracy, not the composite training reward: the run's reward function returns `score = 0.2 * format + 0.8 * acc`, and `acc` is a format-independent correctness rate, so gating on `acc` measures what the training was supposed to buy rather than the output format the policy also learns. `val-core` is produced by the rollouter's own validation pass over all 1,319 test rows at `n=4`, `temperature=1.0`.

Threshold calibration: two runs per side. The threshold is +0.0146 above the defective arm's highest reading at any step in either pass (0.7314), -0.0144 below the lower of the two baseline judged values (0.7604), and +0.0197 above the higher of the two arms' step-0 values (0.7263). Replaying the four recorded runs through this grader at this threshold fails 2/2 defective and passes 2/2 baseline. Two runs per side establish these margins only; no variance is claimed. See `../README.md`.

The verifier writes `experiment_metrics.json`, `metrics.json`, and `result.json`. `test.sh` runs `tests/grade_submission.py`, which reads `resources/grader_spec.json`. It uses no LLM judge at the solution level. `grade_submission.py` is byte-identical to `tasks/algorithm/verl_gsm8k_flat_eval`'s; only the spec differs.

## System rerun (`run_spec.json`)

The solution protocol is debug-then-report: the agent fixes the workspace, submits `/workspace/report.md`, and does **not** launch training. This directory's `run_spec.json` drives the harness rerun stage — the command, environment (with `$PTDBENCH_*` references expanded per backend), timeout, and log roots — executed in the agent's workspace before this verifier runs. The harness deletes the declared log roots (`experiment_logs`) before the rerun, so grading only ever sees this fresh run's logs.

`run_spec.json` pins the budget the protocol requires rather than leaving it to the launch script's defaults: `TOTAL_ROLLOUT_STEPS=1632` (101 trainer steps), `TEST_FREQ=20`, and `VAL_N=4` / `VAL_TEMPERATURE=1.0` for the validation sampling. The timeout is 6 hours, roughly 13x the ~27 min one recorded arm took on 2xA800.
