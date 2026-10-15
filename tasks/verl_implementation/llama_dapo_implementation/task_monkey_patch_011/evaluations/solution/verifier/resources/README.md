# Metric-only solution verifier

`test.sh` extracts the task metric only from fresh `$PTDBENCH_RESULTS/experiment_logs`, including offline W&B files under `$PTDBENCH_RESULTS/experiment_logs/wandb`. It does not grade `submission.json`, code rules, rubrics, or an LLM judgment.

It reads every log under `experiment_logs` and scores the `val-core/math_dapo/acc/mean@1` point at **step 200** (`final_step` in `grader_spec.json`). That value must be at least `0.060` (6.0%). Pass is score `1`; all failures are score `0`. A run whose logs contain no step-200 point fails with `missing ... at step 200`. Earlier steps are parsed and reported (`peak_step` / `peak_score` in `metrics.json`) but are never graded.

The verifier writes `experiment_metrics.json`, `metrics.json`, and `result.json`. `test.sh` runs `tests/grade_submission.py`, which reads `resources/grader_spec.json`.

## Threshold calibration (2026-10-15)

Both numbers in `grader_spec.json` come from a matched pair of runs on devspace
`1177503`, under `/user/chenyinghao/harbor-runs/task_monkey_patch_011__stage__env`.
Same node, same 2 GPUs, same 200 steps, same `TRAIN_BATCH_SIZE=4`, and the
dataset digests `prepare.sh` pins (`test.parquet` 1807 rows after length
filtering). The only difference between the two runs is
`verl/models/transformers/monkey_patch.py`, edited at `04:13:24` between them:

| arm | log | `Monkey patch _flash_attention_forward` | peak | **step 200** |
| --- | --- | --- | ---: | ---: |
| staged defect | `..._20261003_030020.log` | absent | 0.06862 @100 | **0.04759** (86/1807) |
| oracle repair | `..._20261003_041643.log` | present | 0.07471 @180 | **0.07360** (133/1807) |

`required_final_score = 0.060` sits at the midpoint, about 2.2 SE above the
defect arm and 2.4 SE below the repaired arm (binomial SE at p≈0.06, n=1807 is
≈0.0056). `historical_final_baseline` is the defect arm's step-200 value, so
`absolute_improvement` reads as the gain over an unrepaired run.

Two things the previous protocol got wrong, both fixed here:

- **Peak-over-all-steps was graded against `0.07`.** The staged defect run peaks
  at 0.06862 = 124/1807, three correct answers short of the 127 that `0.07`
  requires. An unrepaired workspace was inside sampling noise of passing. At
  step 200 the two arms are 47 answers apart instead of 11.
- **`final_step` was loaded and never used.** Because the max was taken over
  every step, and `TOTAL_TRAINING_STEPS` / `TEST_FREQ` are plain environment
  overrides in the launch script, more steps or a denser eval schedule raised
  the graded number without repairing anything. Grading a fixed step removes
  that lever.

The superseded `0.07` bar and the superseded `historical_final_baseline`
`0.04022346368715084` (72/**1790**) both predate the current dataset recipe;
1790 was the old test split, which `prepare.sh` no longer reproduces.

Known remaining gap: the repaired arm has **one** run, so the step-200
run-to-run variance is unmeasured. A repeat of both arms would firm up the
0.060 bar. One run is ~29 min on 2×H100, well inside the 90 min budget.
