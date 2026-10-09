# Canonical solution oracle

## Root cause

- `run_buggy.sh` (`export LR`)
  - kind: `hparam`
  - observed: `1e-2`
  - expected: `1e-6`
  - note: `LR` flows to the actor optimizer through
    `recipe/vimpo/run_vimpo_critic.sh` as
    `actor_rollout_ref.actor.optim.lr="${LR:-1e-6}"`.

## Observable impact

- Primary source evaluation metric: `val-core/math__math/acc/mean@1`.
- Validation starts non-zero and collapses to zero; response length saturates;
  `actor/grad_norm` explodes. PPO still runs all 8 steps.

## Trigger conditions

- `run_buggy.sh` exports `LR=1e-2`; the recipe default is `1e-6`.

## Required remediation

- Resolve the actor optimizer learning rate to `1e-6` on the real
  launcher-to-optimizer path. Either set `export LR=1e-6` in `run_buggy.sh`, or
  remove the erroneous override and let the recipe default (`${LR:-1e-6}`) apply.
  Do not alter the critic LR, validation data, or the preserved evidence.

Machine grading reads only the new experiment logs. Scoring is weighted:

- validation: the terminal (final-step) `val-core/math__math/acc/mean@1` must be
  at least `required_final_score` (see `grader_spec.json`); the terminal value is
  used, not a peak, because the buggy run peaks early (`~0.69` at step 1) before
  collapsing. The historical final baseline is `0.0`.
- training reward: mean `critic/rewards/mean` over training steps must be
  strictly above `required_training_reward`.
- final `score = 0.8 * validation + 0.2 * training_reward`.

Pass requires both components (`score == 1.0`). The framework still records the
weighted float when only one component passes.

> **Threshold calibration note.** `required_final_score` and
> `required_training_reward` in `grader_spec.json` are evidence-based
> conservative drafts, not values verified against a real healthy rerun. See
> `task.yaml` (`needs_manual_review: true`).
