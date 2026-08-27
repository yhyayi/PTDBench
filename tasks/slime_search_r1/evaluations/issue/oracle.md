# Canonical issue oracle

Reconciled from the task's frozen evidence and canonical verifier record.

Expected verdict: **healthy**.

## Root cause

None. The staged Search-R1 recipe contains no injected defect; the expected
root-cause list is empty.

## Observable impact

- Primary source evaluation metric: `eval/nq_test` (Natural Questions
  exact-match accuracy on the Search-R1 test split).
- Historical cutoff is train step 100. Eval interval is 10; the last held-out
  point in the frozen snapshot is step 99.
- Held-out `eval/nq_test` moves from about 4.6% at step 0 to 37.5%
  (`0.3753`) at step 99. That is a clear, non-collapsing gain.
- Eval response length falls from about 859 to 31 tokens; eval truncation
  falls from about 38% to 0.07%. Train response length at step 100 is about
  39 tokens with a 0% truncate ratio.
- Frozen evidence is in `outputs/metrics.json` and
  `outputs/historical_eval_trace.log`.

## Trigger conditions

- None. The multi-turn search rollout, observation masking, token/log-prob
  alignment, exact-match reward, and GRPO configuration follow the canonical
  Search-R1 reproduction.

## Documentation reconciliation

- Observation (`<information>…</information>`) tokens are appended with
  `loss_mask=0` (`trainable=False`); only model-generated tokens carry
  `loss_mask=1`.
- `max_turns=2`, `topk=3`, `format_score=0.2`, `search_concurrency=256`,
  commented eval/W&B blocks, the local retrieval URL, and `/root/…` asset
  paths are expected recipe properties, not defects.
- A truncated historical log is not by itself proof of a defect. The frozen
  step-100 snapshot is the scored evidence window.
