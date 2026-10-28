# Canonical solution oracle

## Root cause

- `verl/experimental/fully_async_policy/config/fully_async_ppo_trainer.yaml`
  - kind: `hparam`
  - observed: `actor_rollout_ref.rollout` has no `checkpoint_engine.backend`, so
    the `CheckpointEngineConfig` dataclass default applies and the run resolves
    `backend: naive` — the colocated engine, which can only hand weights to a
    rollout object in the same worker process.
  - expected: the stanza pins `checkpoint_engine: {backend: "nccl"}`, an engine
    that can cross Ray resource pools.

- `verl/workers/engine_workers.py`, `ActorRolloutRefWorker.update_weights`
  - kind: `symbol`
  - observed: the colocated branch is preceded by
    `if self.rollout is None: logger.warning(...); return`. Under
    `hybrid_engine=False` the trainer pool is built with
    `train_role = Role.Actor`, so `self.rollout` is `None` on every trainer
    worker and the push returns having transferred nothing, without raising.
  - expected: no silent early return on that path — the guard is removed, or it
    raises.

Both causes are essential. The first sends the push down the colocated path;
the second is why losing it is silent. A diagnosis naming only one earns partial
root-cause credit.

## Observable impact

- Primary source evaluation metric: `val-core/openai/gsm8k/acc/mean@4`.
- The defective run's held-out accuracy does not move across its evaluation
  points; the baseline run's rises over the same points.
- `fully_async/count/current_param_version` climbs identically in both runs, so
  the trainer's own sync counter does not discriminate.
- The `format` component of the reward, read off the per-sample dumps at steps
  1 / 51 / 102, is 0.000 / 0.016 / 0.016 in the defective run and
  0.016 / 0.703 / 0.969 in the baseline: the defective run's rollout policy
  never learns the output format it is rewarded for.

## Trigger conditions

- `actor_rollout_ref.hybrid_engine=False`, so trainer and rollout occupy two Ray
  resource pools and the weight push has to cross between them.
- The recipe omits the backend pin, so `"naive"` is what the run resolves.
- The trainer pool's role is `Role.Actor`, so its workers own no rollout engine.

## Required remediation

- `verl/experimental/fully_async_policy/config/fully_async_ppo_trainer.yaml`:
  restore the baseline's `checkpoint_engine: {backend: "nccl"}` under
  `actor_rollout_ref.rollout`.
- `verl/workers/engine_workers.py`: remove the `if self.rollout is None:`
  early return in `update_weights` (or make it raise), so a backend that cannot
  reach the rollout pool fails loudly.

`baseline/` holds both files as the baseline workspace has them; the diff
against `workspace/` is exactly these two hunks and nothing else.

How the two hunks interact, which is what the two tiers measure differently:

- **Backend pin only.** `update_weights` then takes the
  `backend != "naive"` branch, sends the weights through the checkpoint engine
  and returns before the guard is ever reached. The push works and the metric
  gate passes. The silent path survives, latent, for any future mis-set
  backend — which is the issue tier's second cause, not something the metric
  can see.
- **Guard only.** With `backend` still `"naive"` the call falls through to the
  colocated path. `free_cache_engine` is `False` in this run, so the first
  statement to touch the rollout is
  `await self.rollout.update_weights(...)` on `None`, which raises
  `AttributeError` on the first sync and takes the run down. That is the loud
  variant of this defect, not a repair: it produces no metric and fails the
  gate.
- **Both.** The push works, and a backend that cannot reach the rollout pool
  fails loudly instead of degrading into a no-op.

So the metric gate is satisfied by the backend pin, and the guard is what keeps
the failure mode from recurring silently. A submission that only removes the
guard scores 0 on the metric.

Machine grading reads only the new experiment logs: the binary gate is the
finite `val-core/openai/gsm8k/acc/mean@4` selected by `score_mode`, which must
reach `required_final_score` in `../verifier/resources/grader_spec.json`. No
reference scores, weights or LLM judgement are used at this tier.
