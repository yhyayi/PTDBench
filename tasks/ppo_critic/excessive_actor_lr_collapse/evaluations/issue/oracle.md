# Canonical issue oracle

Reconciled from the task's frozen evidence and canonical verifier record.

Expected verdict: **unhealthy**.

## Root cause

- `run_buggy.sh` (`export LR`)
  - kind: `hparam`
  - observed: `1e-2`
  - expected: `1e-6`
  - note: `LR` is consumed by `recipe/vimpo/run_vimpo_critic.sh` as `actor_rollout_ref.actor.optim.lr="${LR:-1e-6}"`, so the value reaches the actor optimizer. The actor LR is therefore 10,000x the intended target. The critic LR (`CRITIC_LR=1e-5`) is unchanged.

## Observable impact

- the job initializes and completes all 8 PPO steps (not a startup/infra failure)
- `critic/rewards/mean` starts non-zero (≈0.03–0.06) then falls to `0.0`
- `val-core/math__math/acc/mean@1` and `val-core/math__aime_repeated_8x/acc/mean@1` start non-zero and reach zero
- responses saturate the 1024-token `MAX_RESPONSE_LENGTH`
- `actor/grad_norm` and actor entropy become highly unstable

## Trigger conditions

- `run_buggy.sh` overrides `LR` to `1e-2`; the recipe default is `1e-6`.

## Documentation reconciliation

- Exploding `actor/grad_norm`, saturated response length, and unstable entropy are symptoms of the excessive actor LR, not separate root causes. This is an RL optimization failure, not an infrastructure-startup failure.
