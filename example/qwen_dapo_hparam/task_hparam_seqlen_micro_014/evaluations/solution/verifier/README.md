# Solution verifier

This self-contained verifier grades an execution-based solution. It reads
`$PTDBENCH_RESULTS/submission.json` as a run manifest, then validates the
submitted repaired run artifacts.

The verifier is deliberately anchored to the strongest bad-run signals for this
case:

- the effective `MAX_PROMPT_LENGTH` and `MAX_RESPONSE_LENGTH` must be repaired
  to `1024`;
- the repaired job must complete the fixed 200-step protocol;
- the repaired held-out `val-core/math_dapo/acc/mean@1` must recover to the
  healthy range, rather than the frozen bad run's roughly `0.015`;
- if rollout length metrics are present, they must not show the old
  `512`-token response cap with a high clip ratio.

The output is written to `$PTDBENCH_RESULTS/result.json` and
`$PTDBENCH_RESULTS/metrics.json`. This verifier is deterministic and does not
call an LLM judge.
