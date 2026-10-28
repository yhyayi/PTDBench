# Canonical issue oracle

Reconciled from the defective/baseline source comparison and the frozen run logs.

Expected verdict: **unhealthy**.

## Root cause

`verl/experimental/fully_async_policy/config/fully_async_ppo_trainer.yaml` — kind: `hparam`

- **Observed (workspace):** `actor_rollout_ref.rollout` carries no
  `checkpoint_engine.backend`, so the `CheckpointEngineConfig` dataclass default
  applies and the run selects `"naive"` — the *colocated* checkpoint engine,
  which can only hand weights to a rollout object living in the same worker
  process. The resolved config in the frozen log records `backend: naive`.
- **Expected (baseline):** the stanza pins a backend that can cross Ray resource
  pools: `checkpoint_engine: {backend: "nccl"}`.
- **Causal role:** under `"naive"`, `CheckpointEngineManager.update_weights`
  short-circuits after calling the trainer
  (`verl/checkpoint_engine/base.py`), so the vLLM rollout replicas are never
  contacted through the engine at all.

`verl/workers/engine_workers.py` — kind: `symbol`, `ActorRolloutRefWorker.update_weights`

- **Observed (workspace):** the colocated branch is guarded by

  ```python
  # Standalone actor workers have no colocated rollout engine to push into.
  if self.rollout is None:
      logger.warning("update_weights: no rollout engine on role=%s, skipping", self.role)
      return
  ```

  The fully-async trainer pool is built with `train_role = Role.Actor`
  (`verl/experimental/fully_async_policy/fully_async_trainer.py`), so
  `self.rollout` is `None` on every trainer worker and the push returns having
  transferred nothing. No exception is raised, the future resolves, and the
  trainer counts the sync.
- **Expected (baseline):** no silent early return — the guard is removed (or
  raises), so a backend that cannot reach the rollout pool fails loudly instead
  of degrading into a no-op.

Listed causes are all essential. Identifying only part of them must receive
partial root_cause_score.

## Observable impact

- Primary source evaluation metric: `val-core/openai/gsm8k/acc/mean@4`, over all
  1,319 held-out GSM8K rows at `n=4`, `temperature=1.0`.
- The observed run's held-out accuracy does not move across its six evaluation
  points — `0.7259, 0.7195, 0.7227, 0.7314, 0.7257, 0.7248` at steps
  0/20/40/60/80/100, i.e. it ends 0.0011 *below* where the untrained model
  started. The comparison run rises monotonically over the same points,
  `0.7263 -> 0.7364 -> 0.7504 -> 0.7563 -> 0.7690 -> 0.7743`.
- Both arms start from the same untrained model and measure it at essentially
  the same value (0.7259 vs 0.7263 at step 0), so the divergence is produced by
  training, not by the evaluation.
- `workspace/experiment_logs/train.log` contains one
  `update_weights: no rollout engine on role=actor, skipping` warning per
  parameter-version sync — 104 occurrences for 101 trainer steps — and none
  appear in the comparison run.
- `fully_async/count/current_param_version` climbs once per trainer step in
  **both** runs, so the trainer's own sync counter is not evidence of a working
  push.
- The training-side reward is also flat: `critic/score/mean` averages 0.6325
  over the first five trainer steps and 0.6513 over the last five, against
  0.6506 -> 0.8175 in the comparison run.
- The learnable half of that reward is the giveaway. `format` — the share of
  responses that place the answer after `####`, which the prompt asks for and
  which exploration can reach — reads 0.000 / 0.016 / 0.016 in the per-sample
  dumps `workspace/experiment_logs/rollouts/{1,51,102}.jsonl`, while the
  comparison run's same three steps read 0.016 / 0.703 / 0.969. A rollout
  policy generating from frozen weights cannot learn an output format.
- These are recorded runs, not estimates of run-to-run variance. The metric
  curves and the warning count are symptoms/evidence; the configuration default
  and the silent guard are the code-level causes.

## Trigger conditions

- `actor_rollout_ref.hybrid_engine=False`, so the trainer and the rollout engine
  occupy two separate Ray resource pools and a weight push has to cross between
  them. With the two engines colocated the push would land through shared memory
  and the defect would not be reachable.
- The recipe omits the `checkpoint_engine.backend` pin, so the dataclass default
  `"naive"` is what the run resolves.
- The trainer pool's role is `Role.Actor`, so its workers own no rollout engine
  and `self.rollout is None` on the push path.

## Documentation reconciliation

- "Training did not improve" is a **wrong attribution**: the defective run's own
  trainer parameters do keep updating on advantages computed from a stale
  policy's samples, so its reward and its saved checkpoint do drift slightly
  above the untrained model. The claim is that the *rollout* policy is frozen
  and almost none of the training transfers — not that nothing happened.
- `actor/grad_norm` and `actor/ppo_kl` are healthy and overlapping between the
  two runs; neither is a defect.
- The single warning line per sync is the only textual evidence in the log and
  is easy to miss; its absence in the comparison run is what makes it evidence.
- Colocating the two engines (`hybrid_engine=True`) or enabling
  `async_training.use_trainer_do_validate` would both make the symptom
  disappear by changing the topology rather than repairing the push, and neither
  is a correct diagnosis of this run.
- The vendored tree also carries a `None`-tolerance workaround in
  `verl/experimental/fully_async_policy/detach_utils.py` (weight versions that
  come back `None` are read as 0 so the partial-rollout statistics stay
  computable). It is byte-identical in **both** runs and is what lets this
  recipe complete at all on this release; it is not part of the defect.
