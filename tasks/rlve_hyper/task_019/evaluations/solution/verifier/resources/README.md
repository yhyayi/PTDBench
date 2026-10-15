# Metric-only solution verifier

`test.sh` is the single entry point. It does not use `submission.json`, an LLM,
rules, or rubrics. It strips ANSI escapes and reconstructs metrics from
`$PTDBENCH_RESULTS/experiment_logs`, including offline W&B files under
`$PTDBENCH_RESULTS/experiment_logs/wandb`.

It reads every log under `experiment_logs` and assigns a weighted score:

- validation: `eval/HELD-OUT_ENVIRONMENTS_128` **at step 99** (`final_step`, the
  last step of the locked 100-rollout budget) must exist and fall in
  `[-0.40, 0.30]` to score `1`, otherwise `0`. At least `min_eval_points` (2)
  eval points must be present.
- learning rate: `train/lr-pg_0` must be logged at step 99, cover at least
  `min_learning_rate_points` (95) steps, and its mean over training steps must
  be at least `8e-7` to score `1`, otherwise `0`
- final `score = 0.8 * validation + 0.2 * learning rate`

Pass requires both components (`score == 1.0`). The framework still records
the weighted float when only one component passes.

The verifier writes `experiment_metrics.json`, `metrics.json`, and
`result.json`. `test.sh` runs `tests/grade_submission.py`, which reads
`resources/grader_spec.json`.

## Calibration

Both thresholds are calibrated against two reruns of this exact staged
workspace on 4xH100 (`--num-rollout 100`, `--eval-interval 20`,
`--n-samples-per-eval-prompt 1`, `--eval-top-p 0.7`, 128 held-out prompts).
The two trees differ only by the oracle repair in
`scripts/training/Qwen2.5-0.5B-Instruct/rlve.sh`.

| step | defect arm (`--lr-warmup-iters 2000`) | repaired arm (`--lr-warmup-iters 0`) |
| ---: | ---: | ---: |
| 0 | -0.6195 | -0.5767 |
| 19 | -0.6149 | -0.2533 |
| 39 | -0.5635 | -0.2384 |
| 59 | -0.6305 | -0.1355 |
| 79 | -0.6525 | -0.1419 |
| **99** | **-0.5321** | **-0.1021** |
| mean `train/lr-pg_0` | 5.05e-8 | 2.0e-6 |

Noise scale: both arms evaluate the *same* base checkpoint at step 0 and differ
by 0.042, so one eval point carries sigma ~= 0.03.

- `required_final_score = -0.40` is placed between the defect arm's step-99
  value (-0.5321) and the repaired arm's *worst* post-training point (-0.2533 at
  step 19), whose midpoint is -0.393. That is ~0.13 of margin on each side. The
  repaired arm's own post-training points swing by 0.15, so the bar is set
  against that worst point rather than against the step-99 value alone -- a
  midpoint of the two step-99 values (-0.317) would leave only 0.047 of headroom
  below the repaired arm's weakest checkpoint. The previous `-0.24` was a family
  constant shared verbatim by `task_{001,012,019}` despite step-99 baselines of
  -0.639 / -0.614 / -0.522, and it predates both reruns.
- `required_final_score_max = 0.30` is a plausibility ceiling. The RLVE verifier
  returns a reward in `[-1, 1]` (`Gym/environment.py:183`), so the metric is a
  bounded mean; the repaired arm finished at -0.102, and nothing a 100-step
  0.5B GRPO run can do reaches +0.30. A value above it means fabricated lines or
  an off-protocol eval set, which the deterministic grader cannot otherwise see.
- `required_learning_rate = 8e-7` separates the two measured means by 16x on the
  defect side and 2.5x on the repaired side. It is left where it was because
  both measured arms are far from it.
- `historical_final_baseline` (-0.5221, from the original full-scale run) is
  confirmed by the staged defect rerun's -0.5321, so `absolute_improvement`
  stays meaningful even though the staged workspace is downscaled
  (`n_samples_per_prompt` 8 vs 16, `rollout_max_response_len` 2048 vs 8192).

## Why step 99 and not the peak

Earlier revisions graded the highest value at any step. Two problems, both
measured:

- A repaired run truncated after `eval 39` scored `1.0` on a 40-step budget
  (its step-39 value, -0.2384, cleared the old `-0.24` bar by 0.0016), so 60% of
  the compute budget was optional.
- Peak-over-all-steps grows with the budget, so raising `--num-rollout` was a
  free way to buy a better number. A fixed step index is immune to that: extra
  steps past 99 are simply ignored.

Residual, accepted: values are still merged per step across every log file under
`experiment_logs` with a per-step max, so repeated runs give best-of-N at step
99. `oracle/task.yaml` declares `max_runs: 1` but nothing in the container
enforces it.

## Wall clock

The repaired rerun took 5h28m end to end on 4xH100 (02:19:51 -> 07:48:21, a
steady ~65 min per 20 steps, no stalls). Validation now requires the step-99
point, so the agent needs all of it. `readme.md`'s preserved
`verify.timeout_minutes: 180` is roughly half that, but it is documentation
only: `harbor_evaluator._write_task_config()` emits no `[agent] timeout_sec`, so
the agent phase is untimed in practice.

## Residual: partial repairs

A repair that only shortens the warmup instead of removing it (say
`--lr-warmup-iters 100`) keeps mean `train/lr-pg_0` at ~1e-6, which clears the
`8e-7` learning-rate component, and could land its step-99 eval between the two
arms. Where the bar sits is a trade: -0.40 protects a genuine repair against
run-to-run variance, at the cost of possibly passing a half-repair. No partial
arm has been measured, so this is unquantified.
