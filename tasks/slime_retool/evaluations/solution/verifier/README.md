# Metric-only solution verifier

`test.sh` is the single entry point. It does not use `submission.json`, an LLM,
rules, or rubrics. It strips ANSI escapes and reconstructs `eval/aime` from
`$PTDBENCH_RESULTS/experiment_logs`, including offline W&B files under
`$PTDBENCH_RESULTS/experiment_logs/wandb`.

`eval/aime` is read in the math_dapo **score** space, the range `[-1, 1]` the
frozen historical trace was logged in. This recipe's `reward_func` is
unmodified, so the score channel is already the exact-match signal and no
`--eval-reward-key` is needed to get at it — unlike `slime_retool_length_hack`,
where the score channel carries an injected length bonus and the instruction
has to ask for `acc`. Grading in the same space the history was logged in is
what makes the threshold below directly comparable to it.

`score_mode` is `best`: the verifier takes the highest finite `eval/aime` over
the run rather than the value at step 69. The bar here is "the run actually
trained", and a single eval point over 30 AIME problems is noisy enough that
the last one is a poor proxy for that.

That value must be at least `-0.55` and at most `1.0`. Pass is score `1`; all
failures are score `0`. The threshold is calibrated against the 8 points in
`workspace/outputs/metrics.json`: an untrained step 0 sits at `-0.5808` and
fails, and because `score_mode` is `best`, early trained points that are still
below the bar (step 9 is `-0.5706`) do not sink a run that later learns. The
run clears the bar comfortably — its best point is step 59 at `-0.4246`, 0.125
clear — and even the last point (step 69) is `-0.4861`, about 25.7% accuracy.

The verifier writes `experiment_metrics.json`, `metrics.json`, and
`result.json`. `test.sh` runs `tests/grade_submission.py`, which reads
`resources/grader_spec.json`.
