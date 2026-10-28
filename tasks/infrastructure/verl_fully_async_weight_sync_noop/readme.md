# Silent Weight-Sync No-Op in Fully-Async VERL GRPO

Training Qwen3-1.7B with fully-async GRPO on GSM8K, with the trainer and the
vLLM rollout engine in two separate Ray resource pools. Every weight push from
the trainer to the rollout engine is dropped silently, so the policy that
generates rollouts stays at its initial weights for the whole run.

| | |
| --- | --- |
| **Author** | czx |
| **Type** | infrastructure |
| **Model** | Qwen/Qwen3-1.7B |
| **Dataset** | OpenAI GSM8K main — 7,473 train rows (1,632 consumed, one partial epoch, no prompt seen twice) / 1,319 held-out test rows |
| **Framework** | verl, `run.sh` |
| **Environment** | `verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12` · 2 GPUs |
| **Evaluations** | issue · solution |
| **Recorded runs** | 2 defective + 2 baseline runs, 101 trainer steps each, ~27 min per run |

## Difficulty

The defect needs two files to be silent. The recipe yaml
`verl/experimental/fully_async_policy/config/fully_async_ppo_trainer.yaml`
carries no `checkpoint_engine.backend` under `actor_rollout_ref.rollout`, so the
`CheckpointEngineConfig` dataclass default applies and the run resolves
`backend: naive` — the *colocated* engine, which can only hand weights to a
rollout object inside the same worker process. Then, in
`verl/workers/engine_workers.py`:

```python
# 0. send_weights only for async training with disaggregated trainer and rollout
if self.config.rollout.checkpoint_engine.backend != "naive":
    per_tensor_param, _ = self.actor.engine.get_per_tensor_param()
    await self.checkpoint_engine.send_weights(per_tensor_param)
    return

# Standalone actor workers have no colocated rollout engine to push into.
if self.rollout is None:
    logger.warning("update_weights: no rollout engine on role=%s, skipping", self.role)
    return
```

Under `hybrid_engine=False` the trainer pool is built with
`train_role = Role.Actor`, so `self.rollout` is `None` on every trainer worker.
The first branch is skipped because the backend *is* `"naive"`; the second
returns. No exception is raised, the future resolves, and the trainer counts the
sync. `CheckpointEngineManager.update_weights` has already short-circuited
above this, so the rollout replicas are never contacted either.

Nothing fails loudly. The run completes all 101 trainer steps with no
traceback, `actor/grad_norm` and `actor/ppo_kl` healthy and overlapping with the
baseline's, and `fully_async/count/current_param_version` climbing once per step
in both arms — so the trainer's own sync counter, the natural thing to check, is
the strongest distractor in the task. The entire textual evidence is one
`update_weights: no rollout engine on role=actor, skipping` warning per sync in
a multi-megabyte driver log.

The single strongest piece of evidence is the `format` component of the reward,
which separates the arms by a factor of 60 while accuracy separates them by 5 pp:
0.000 / 0.016 / 0.016 at steps 1 / 51 / 102 of the defective run against
0.016 / 0.703 / 0.969 in the baseline. Placing the answer after `####` is
reachable by exploration and is exactly what a policy learns when it is being
updated, so a rollout policy that never learns it is generating from weights
that never changed. That is a behavioural measurement of the frozen policy and
no amount of error handling moves it.

A weight-digest probe — hashing a fixed slice of the live vLLM parameters from
outside the job — was used while building this task and is the most direct
evidence of all, but it is deliberately **not** shipped: a tool in the workspace
whose only purpose is to detect this defect would hand the agent the diagnosis.

The wrong diagnosis the task is built to reject is "training did not improve":
the trainer's own parameters do keep updating, on advantages computed from a
stale policy's samples, so the defective run's reward and its saved checkpoint
drift slightly above the untrained model. The claim is that the *rollout* policy
is frozen and almost none of the training transfers.

`val-core/openai/gsm8k/acc/mean@4`, all 1,319 held-out rows at `n=4`,
`temperature=1.0`, two runs per side:

| step | defective | baseline |
| --- | --- | --- |
| 0 | 0.7259 / 0.7221 | 0.7263 / 0.7235 |
| 20 | 0.7195 / 0.7208 | 0.7364 / 0.7360 |
| 40 | 0.7227 / 0.7263 | 0.7504 / 0.7468 |
| 60 | 0.7314 / 0.7271 | 0.7563 / 0.7540 |
| 80 | 0.7257 / 0.7286 | 0.7690 / 0.7530 |
| 100 | **0.7248 / 0.7199** | **0.7743 / 0.7604** |

Both arms measure the same untrained model at step 0 to within 0.4 pp, so the
divergence is produced by training rather than by the evaluation. The defective
arm ends 0.1 pp *below* its own starting point in pass 1 and 0.2 pp below in
pass 2; its highest reading at any step in either pass is 0.7314.

The training-side reward tells the same story from the other end: `format`, the
share of responses that put the answer after `####`, reads 0.000 / 0.016 / 0.016
at steps 1 / 51 / 102 of the defective run and 0.016 / 0.703 / 0.969 at the same
steps of the baseline — a frozen rollout policy cannot learn an output format.

## Reference solution

- **issue**: a passing diagnosis must name both the missing backend pin (the run
  resolving to the colocated `"naive"` engine) and the
  `if self.rollout is None: ... return` guard that makes losing the push silent,
  state that the trainer pool owns no rollout engine so the colocated path has
  no destination, and support it with the flat held-out curve, the per-sync
  warning, or the flat `format` component of the reward. Naming only one of
  the two causes
  earns partial root-cause credit. Attributing the symptom to the two-pool
  topology itself, or proposing to colocate the engines, must fail.
- **solution**: pin `checkpoint_engine.backend: "nccl"` in the recipe yaml and
  delete the `self.rollout is None` early return (or make it raise) — 8 lines
  across two files. The pin alone satisfies the metric: `update_weights` then
  takes the `backend != "naive"` branch and returns before the guard is reached.
  Removing the guard alone does **not** pass — with `backend` still `"naive"`
  the call falls through to `await self.rollout.update_weights(...)` on `None`
  and the run dies on the first sync, which is the loud variant of this defect.

## Verification

- **issue**: an LLM judge compares `agent_answer.md` with the hidden oracle on
  verdict / root cause / evidence, weighted 0.2 / 0.6 / 0.2. Passing requires
  ≥ 0.8.
- **solution**: the harness re-runs `run.sh` in the repaired workspace and grades
  `val-core/openai/gsm8k/acc/mean@4` against `required_final_score` = `0.746`
  with `score_mode: best` and `final_step: 100`. Calibrated on two runs per
  side: the threshold is **+0.0146** above the defective arm's highest reading
  at any step in either pass (0.7314), **-0.0144** below the lower of the two
  baseline runs' judged values (0.7604), and **+0.0197** above the higher of the
  two arms' step-0 values (0.7263), so repairing the code without completing the
  run cannot pass. Both arms were replayed through the shipped grader at this
  threshold: 2/2 defective fail, 2/2 baseline pass, under both `best` and
  `final`. Two runs per side establish these margins only, not a variance
  estimate.

### Not verified here

- The harness end-to-end path (`python3 -m ptdbench run`) and
  `./docker/warm-cache.sh` were not exercised: this machine's Docker host cannot
  reach Docker Hub, so the environment image had to be moved in by hand and the
  prepare stage was never driven by the harness itself. The four recorded runs
  were launched directly against the pinned image with the workspace, the
  prepared assets and `$PTDBENCH_RESULTS` mounted the way the harness mounts
  them, and `run_spec.json`'s overrides applied; `environment/prepare.sh` was
  validated by argument parsing and by building the GSM8K recipe inside the
  task's image, not by a full fetch from scratch.
- Asset checksums: the model file hashes were read back from the pinned
  revision through hf-mirror; the weights themselves were not re-downloaded.
  Both dataset `--expect` values are environment-dependent — they were taken
  from inside this task's image and verified reproducible by building twice
  there, but a different pandas writes different bytes for identical rows, so
  they must be re-read if the image is repinned.
- Two runs per side establish the threshold's margins only. No variance or
  statistical significance is claimed, and the defective arm's exposure under
  `score_mode: best` is bounded by 12 observed evaluation points rather than by
  a distribution.
- The issue tier's LLM judge was not run: it needs `PTDBENCH_JUDGE_MODEL` and an
  OpenAI-compatible endpoint. Only the shared protocol files' byte-identity and
  the oracle's factual claims were checked.
