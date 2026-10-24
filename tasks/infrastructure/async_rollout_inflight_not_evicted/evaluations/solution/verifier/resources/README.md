`test.sh` is the single entry point. It does not use `submission.json`, an LLM,
rules, or rubrics. It runs `tests/grade_submission.py`, which reads
`resources/grader_spec.json` and scores only from
`$PTDBENCH_RESULTS/experiment_logs`.

Two signals, both required to pass:

- `step_metric_key`: `train_steps` — the largest `rollout N:` index + 1 in the
  driver log. `required_train_steps`: `120` (defective 92, control 148).
- `solve_rate_metric_key`: `rollout_solve_rate_last_window` — mean fraction of
  samples with `reward.acc` over the last `solve_window_frac` (`0.1`) of dumped
  rollout steps (`dump/rollout_data/*.pt`). `required_solve_rate`: `0.48`
  (defective 0.460, control 0.539).

Pass is score `1` (both thresholds cleared); any shortfall or missing signal is
score `0`. Weights (`step_weight` 0.8, `solve_rate_weight` 0.2) are recorded for
reference; the gate is conjunctive (both must pass), not a weighted sum.
