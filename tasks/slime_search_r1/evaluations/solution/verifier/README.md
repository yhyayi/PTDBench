# Metric-only solution verifier

`test.sh` is the single entry point. It does not use `submission.json`, an LLM,
rules, or rubrics. It strips ANSI escapes and reconstructs `eval/nq_test` from
`$PTDBENCH_RESULTS/experiment_logs`, including offline W&B files under
`$PTDBENCH_RESULTS/experiment_logs/wandb`.

It reads every log under `experiment_logs` and scores the highest finite
metric at any step. That peak must be at least `0.28` (28%). Pass is score
`1`; all failures are score `0`. A later lower step does not fail the run if
an earlier peak already meets the threshold.

`score_mode` is `best` for the same reason as everywhere else in this family:
the bar is "the run actually trained", and a single eval point is a noisy
proxy for that.

The threshold is the one `slime_search_r1_nomask` uses, deliberately: the two
tasks are the healthy/injected pair on the same recipe, the same metric and the
same 100-step horizon, so a differently calibrated bar would be a tell in the
task package. It is well placed for this side of the pair too. Calibrated
against the 11 points in `workspace/outputs/metrics.json`:

- an untrained step 0 sits at `0.046` and fails;
- the historical trace crosses the bar at step 49 (`0.3007`) and peaks at
  `0.3763` (step 89), so an unmodified run clears it with `0.096` of margin;
- the injected twin's broken run peaks at `0.1527` — so an agent that invents a
  defect and "fixes" this healthy recipe into that failure mode (training on
  retrieved `<information>` tokens) fails the level, which is the point.

The historical step-99 value is `0.3753`, about 37.5% accuracy.

The verifier writes `experiment_metrics.json`, `metrics.json`, and
`result.json`. `test.sh` runs `tests/grade_submission.py`, which reads
`resources/grader_spec.json`.

`gold.yaml` is reviewer-facing documentation of the canonical verdict. This
verifier does not read it.
