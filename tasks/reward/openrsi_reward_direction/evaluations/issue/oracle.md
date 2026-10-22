# Canonical issue oracle

Reconciled from the task's frozen evidence and canonical verifier record.

Expected verdict: **unhealthy**.

## Root cause

The score-to-reward mapping no longer honours the per-task `higher_is_better`
flag. In `examples/openmle_rl/reward_func_utils.py`, a helper
`_task_direction(meta)` was introduced that unconditionally returns `True`, with
a comment claiming that all OpenMLE metrics share a normalised scale and
therefore need no per-task direction. Both direction consumers now call it:

- `_signed(score, meta)` — used to place a raw score in signed space;
- `_static_bound_limits_signed(meta)` — used to build the signed
  `(best, worst)` bound pair and to choose which of `leaderboard_min` /
  `leaderboard_max` is best.

Because `_task_direction` always returns `True`, both treat every metric as a
maximisation target.

## Observable impact

The task pool used by this run is deliberately mixed on direction: roughly half
of the MLE tasks are lower-is-better (RMSLE, log-loss, RMSE), the rest are
accuracy-like. With direction ignored:

- For a lower-is-better task, a *better* submission (lower error) receives a
  *lower* reward than a worse one. The advantage ordering within the group
  inverts — the best sample carries a negative advantage and the worst carries a
  large positive one — so the policy is reinforced for making those metrics
  worse.
- For a maximisation task nothing changes, because `_signed` already returned the
  raw score in that case.

The signature is therefore **heterogeneous**: approximately half the task pool
degrades monotonically through training while the other half trains normally.
Aggregate reward can still look healthy because the maximisation half masks the
inversion. Held-out evaluation is the discriminator: the lower-is-better
evaluation tasks get worse while the rest improve.

Note the two near-miss mutations that must not be confused with the root cause:
weakening only `_signed` (or only the bound side) saturates every sample to a
single reward and zeroes the advantage instead of inverting it.

## Trigger conditions

- A task whose metadata has `higher_is_better` false (RMSLE / log-loss / RMSE).
- `USE_SCORE2REWARD` enabled with `REWARD_MAPPING_MODE="power_clip"` (the
  launcher default), so `score2reward` runs through `_signed` and the bound pair.
- The mixed-direction task pool, so that both behaviours are observable in one
  run.

## Documentation reconciliation

- A parent-relative operator shaping rule (draft / improve / debug / crossover)
  and the delta bonus are expected recipe behaviour, not defects.
- `ADAPTIVE_REWARD_BOUND_MODE`, the adaptive bound pair, and the entropic
  advantage are expected properties of the recipe, not defects.
- A log that ends early, or a reward that merely looks noisy, is not by itself
  proof of a defect.
