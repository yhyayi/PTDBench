# Canonical issue oracle

Reconciled from the task's frozen evidence and canonical verifier record.

Expected verdict: **unhealthy**.

## Root cause

- `examples/search-r1/generate_with_search.py::generate`
  - kind: `symbol`
  - observed: retrieved `<information>…</information>` observation tokens are appended with `loss_mask += [1] * len(obs_tokens_ids)`, and that local list is assigned to `sample.loss_mask`, so search-engine results are trained on
  - expected: mask observation tokens out of the loss (`loss_mask += [0] * len(obs_tokens_ids)`), matching the paired `append_response_tokens(..., trainable=False)` call; only model-generated tokens should carry `loss_mask=1`

Naming this cause with the correct causal role is a complete attribution.
Do not require additional causes.

## Observable impact

- Primary source evaluation metric: `eval/nq_test`.
- Historical cutoff is train step 100. Eval interval is 10; the last held-out
  point in the frozen snapshot is step 99.
- Frozen `eval/nq_test` reconstructs ten held-out points that rise from 4.3%
  at step 0 to 14.5% (`eval_last` 0.145; peak 15.3% at step 89). Eval step 19
  is missing from captured stdout; do not invent it.
- A healthy Qwen2.5-3B Search-R1 GRPO run under the same recipe, with
  observation tokens masked (`loss_mask=0`), reaches about **37.5%**
  (`eval/nq_test` 0.3753) at the same 100-step horizon.
- Training `rollout/raw_reward` at step 99 is 0.1875 and is not a substitute
  for held-out health. Truncation falls from ~39% at step 0 to ~0.1% by
  step 99 while held-out accuracy stays far below the healthy protocol.
- Observation tokens also carry dummy `rollout_log_probs` of `0.0`, so
  `loss_mask=1` on those tokens optimizes against meaningless reference log
  probs and corrupts the GRPO objective.
- Frozen evidence is in `outputs/metrics.json` and
  `outputs/historical_eval_trace.log`.

## Trigger conditions

- `generate` appends retrieved observation tokens with `loss_mask=1` and
  assigns that list to `sample.loss_mask`.
- The paired `append_response_tokens(..., trainable=False)` call still marks
  those tokens non-trainable, but the local list is the one that wins.

## Documentation reconciliation

- The comment next to the dummy log-prob fill still says the observation
  tokens "won't be used due to loss_mask=0"; that comment is stale relative
  to the `loss_mask += [1]` assignment.
- `max_turns`, `topk=3`, `format_score=0.2`, `search_concurrency=256`, the
  local retrieval URL, and the GRPO hyperparameters are recipe
  configuration, not the scored cause.
- A diagnosis that names the observation loss mask with the correct causal
  role should receive full `root_cause_score`.
