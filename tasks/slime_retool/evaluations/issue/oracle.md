# Canonical issue oracle

Reconciled from the task's frozen evidence and canonical verifier record.

Expected verdict: **healthy**.

## Root cause

None. The staged ReTool recipe contains no injected defect; the expected
root-cause list is empty.

## Observable impact

- Primary source evaluation metric: `eval/aime` (mean of `reward_func` scores
  on AIME 2024). Accuracy is `(score + 1) / 2`.
- Historical cutoff is train step 100. Eval interval is 10; the last held-out
  point in the frozen snapshot is step 99.
- Held-out `eval/aime` moves from about −0.591 (20.4% acc) at step 0 to
  −0.497 (25.2% acc) at step 99. That is a modest, non-collapsing gain.
- Eval response length grows from about 4.2k to 10.1k tokens; eval truncation
  rises from about 1% to 22%. Train response length at step 100 is about 6.4k
  with a 5.1% truncate ratio. Length growth is present but remains bounded.
- Frozen evidence is in `outputs/metrics.json` and
  `outputs/historical_eval_trace.log`.

## Trigger conditions

- None. The multi-turn tool rollout, observation masking, token/log-prob
  alignment, math_dapo exact-match reward, and GRPO configuration follow the
  canonical ReTool reproduction.

## Documentation reconciliation

- Reward is `math_dapo_compute_score(..., strict_box_verify=True)` plus a
  bounded tool-call shaping term that only nudges already-failing samples.
  Generation length does not add reward.
- `kl-loss-coef 0.00`, `--rotary-base 5000000`, `max_turns=16`, W&B debug
  logging, and `/root/…` asset paths are expected recipe properties, not
  defects.
- A truncated historical log is not by itself proof of a defect. The frozen
  step-100 snapshot is the scored evidence window.
