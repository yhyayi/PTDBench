# Metric-only solution verifier

`test.sh` reads only fresh logs under `$PTDBENCH_RESULTS/experiment_logs`, including text logs from the foreground 50-step rerun. It extracts `val-core/openai/gsm8k/reward/mean@1` at step 50 and passes when the finite score is at least `0.74609375`.

The threshold was calibrated from one observed run per arm: the defective arm's maximum was `0.734375`, the baseline arm's minimum was `0.75390625`, and the baseline workspace's step-0 score was `0.73828125`. The threshold is `0.01171875` above the defective maximum, `0.0078125` below the baseline minimum, and `0.0078125` above step 0. These margins describe the recorded runs only; one run per arm does not establish variance or statistical significance.

The verifier writes `experiment_metrics.json`, `metrics.json`, and `result.json`. It uses no LLM judge at the solution level.
