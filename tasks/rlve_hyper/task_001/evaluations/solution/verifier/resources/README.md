# Metric-only solution verifier

`test.sh` is the single entry point. It does not use `submission.json`, an LLM,
rules, or rubrics. It strips ANSI escapes and reconstructs
`eval/HELD-OUT_ENVIRONMENTS_128` from `$PTDBENCH_RESULTS/experiment_logs`,
including offline W&B files under `$PTDBENCH_RESULTS/experiment_logs/wandb`.

It grades the value at `final_step` (99) only. Pass is score `1`; all failures
are score `0`. A missing step 99 is a failure, so a run that is killed early
cannot score. The curve peak is still reported as `peak_step` / `peak_score`
but never graded.

The verifier writes `experiment_metrics.json`, `metrics.json`, and
`result.json`. `test.sh` runs `tests/grade_submission.py`, which reads
`resources/grader_spec.json`.

## Calibration

Both arms were run under this exact protocol (4x H100, `--num-rollout 100`,
`--eval-interval 20`, so eval lands on steps 0/19/39/59/79/99) on devspace
1107990 under `/user/chenyinghao/harbor-runs/task_001__stage__env`. They differ
only in `scripts/training/Qwen2.5-0.5B-Instruct/rlve.sh::--attention-dropout`:

| step | 0 | 19 | 39 | 59 | 79 | **99** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| repaired (`--attention-dropout 0`) | -0.662 | -0.287 | -0.177 | -0.215 | -0.386 | **-0.170** |
| defect (`--attention-dropout 0.5`) | -0.648 | -0.645 | -0.656 | -0.600 | -0.640 | **-0.634** |

`required_final_score` is `-0.50`, not the midpoint of the two step-99 values
(-0.402). The two arms have very different spread, so the bar is placed by
spread rather than by midpoint:

- The defect arm never learns. Its six points span `[-0.656, -0.600]`
  (mean -0.637, sd 0.020), so `-0.50` sits about 7 sd above the whole curve and
  rejects it even at its single best point (-0.600, which fails by 0.100).
- The repaired arm swings by 0.22 across checkpoints (-0.386 at step 79 to
  -0.170 at step 99) and only one trajectory has been observed, so headroom on
  this side is the scarcer resource. `-0.50` clears the repaired arm's *worst*
  post-training point by 0.114, so an unlucky final step still passes; the
  observed step 99 clears by 0.465.

`historical_final_baseline` is the defect arm's step-99 value from that staged
rerun (-0.6343358521595779), not the step-59 value of the original source run
(-0.6393607996231563). The source run stopped after 4 eval points, so it has no
step 99 to compare against; `absolute_improvement` is only meaningful against
the same 100-rollout schedule the agent is asked to run.

Open: only one repaired trajectory exists, so step-99 variance across reruns is
unmeasured. `-0.50` tolerates the full within-run spread of that trajectory,
which is the best available proxy.
