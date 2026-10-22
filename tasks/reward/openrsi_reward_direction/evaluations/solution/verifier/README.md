# Solution verifier

`test.sh` is the single entry point. It does not use `submission.json`, an LLM,
rules, or rubrics. It strips ANSI escapes and reconstructs the held-out metric
from `$PTDBENCH_RESULTS/experiment_logs`, including offline W&B files under
`$PTDBENCH_RESULTS/experiment_logs/wandb`.

It reads every log under `experiment_logs` and scores the highest finite value
of `metric_key` at any step. That peak must be at least `required_final_score`.
Pass is score `1`; all failures are score `0`. A later lower step does not fail
the run if an earlier peak already meets the threshold.

`tests/grade_submission.py` is reused byte-for-byte from
`tasks/slime_search_r1` (sha256
`6e54036595ee303aa24b1803311b9c9ca91dcced88c146b09962e27ced485215`), so only
this spec differs.

## PLACEHOLDER — thresholds not yet calibrated

`grader_spec.json` currently carries placeholder values and the task must not be
shipped until they are measured on a real run:

- `metric_key` must be the held-out metric the experiment runner actually
  records (currently `eval/validation_tts_draft176/medal_rate`, the headline
  `eval/{key}-avg_score` key that `patch_eval_logger.py` actually emits).
- `final_step` must match the pinned `--num-rollout` in
  `evaluations/solution/instruction.md`.
- `historical_final_baseline` must be the healthy run's value at `final_step`.
- `required_final_score` must be calibrated from the pair: below the healthy
  run's peak and above the injected run's peak, with an explainable margin.

`score_mode` is `best` for the same reason as the rest of this family: the bar
is "the run actually trained", and a single eval point is a noisy proxy. Note
that the held-out evaluation is aggregated over the MLE-Bench Lite CPU subset
with repeated runs per eval point, so the metric being read is already an
average rather than a single stochastic sample.

`gold.yaml` is reviewer-facing documentation of the canonical verdict. This
verifier does not read it.
