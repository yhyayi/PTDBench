# task_dapo_math_021 — GRPO loses its group baseline

Maintainer-facing. Not exposed to the agent.

## Case overview

| | |
| --- | --- |
| evaluation key | `val-core/math_dapo/acc/mean@1` |
| family | `verl_implementation/llama_dapo_implementation` |
| Lark 分类 | ③ Algorithm |
| schema version | 2 |
| levels | `issue` only (see "Why issue-only for now") |
| model | Llama-3.2-1B-Instruct |
| dataset | DAPO-Math-17k |
| framework | verl `0.9.0.dev` @ `6f2aa24abfc01cc72811b376aeb3936503885096` |
| algo | GRPO |
| launch script | `examples/grpo_trainer/run_llama3_2_1b_instruct_dapo_math_fsdp.sh` |
| GPUs | 2 |
| budget | 60 steps, eval every 20 |

## Canonical failure reason

The per-prompt `uid` assignment in
`verl/trainer/ppo/ray_trainer.py::RayPPOTrainer.fit` was moved below
`batch = batch.repeat(repeat_times=rollout.n, interleave=True)`, so each of
the `train_batch_size * rollout_n` sampled responses receives its own fresh
UUID instead of inheriting its prompt's. `uid` is the grouping key that
`compute_advantage` hands to `compute_grpo_outcome_advantage` as `index`, so
every GRPO group becomes a singleton. That function's `len(group) == 1` branch
silently substitutes `id2mean = 0.0` and `id2std = 1.0` rather than raising,
so the group baseline is never subtracted and the per-group standard
deviation never divides: GRPO degenerates into REINFORCE with no baseline over
the raw 0/1 reward.

## Workspace and hidden evaluations

`workspace/` is the defect-side snapshot: upstream verl at the pinned commit
with one file modified, plus the recipe's launch script and reward wrapper.
`evaluations/issue/oracle.md` is hidden from the agent and is what the LLM
judge actually reads.

**`environment/` is mounted read-only into the agent container** at
`/ptdbench/environment` (`ptdbench/agents/base.py`), so it holds only
`prepare.sh`. `validation/source_manifest.json` — which spells out the
injected defect — deliberately lives under `validation/` instead of the
`environment/` location that `skills/package-task/SKILL.md` suggests.
`validation/` is never mounted. Do not move it back.

## Source and preserved context

Base commit: `6f2aa24abfc01cc72811b376aeb3936503885096` (2026-07-15), the
same base as the sibling task `task_monkey_patch_011`, whose recorded
2026-07-20 run executed on the `verl` PTDBench image. That is the evidence
this commit is compatible with the image; the two trees differ only by each
task's own injected defect plus the sibling's bench scaffolding.

Filtered root-cause diff — exactly one file, one relocation:

```text
verl/trainer/ppo/ray_trainer.py
  -  uid assignment above  `gen_batch = self._get_gen_batch(batch)`
  +  uid assignment below  `batch = batch.union(gen_batch_output)`
```

The move is verbatim, comment included, so the staged tree still has exactly
one uid assignment in the training loop and two `uuid.uuid4()` call sites in
the file, as upstream does. An agent cannot find this by counting
assignments or by noticing a stray extra statement; only the position
relative to `batch.repeat` is wrong. Full provenance, including what was
deliberately *not* carried over from the sibling, is in
`validation/source_manifest.json`.

## Original failure evidence

Recorded 2026-10-16 on a Cybertron devspace (job `1250006`, cluster `beijing`,
2× NVIDIA H800), 60 steps, exit code 0, ~13.5 min wall clock. Shipped in
`workspace/logs/`; see that directory's `README.txt` for which file to trust.

**Both arms were run**, identically except for the uid relocation — the
control arm's `ray_trainer.py` is byte-identical to pristine upstream
(sha256 `e636912a…`). The control-arm logs are in
`validation/control_arm_reference/` (maintainer-side, never mounted).

| metric | defect arm | control arm (= upstream) |
| --- | --- | --- |
| steps with `critic/advantages/min` < 0 | **0 / 60** | 29 / 60 |
| most negative `critic/advantages/min` | **0.000000** | −0.935413 |
| distinct nonzero `critic/advantages/max` | **one value: `0.9999990463`** | three: `1.2076`, `1.6202`, `2.4749` |
| `actor/pg_loss` range | **[−0.0674, 0]**, never positive | [−0.0139, +0.0366] |
| `actor/entropy` first → last | 4.81 → **1.31** | 4.42 → 2.99 |
| `val-core/…/acc/mean@1` @ 0/20/40/60 | .0454 / .0437 / .0476 / .0570 | .0454 / .0542 / .0504 / .0592 |
| rewarded steps | 33 / 60 | 29 / 60 |

## Canonical expected outcome

Verdict: **unhealthy**. Root cause as above. The fix is to move the uid
assignment back above `self._get_gen_batch(batch)`.

Two measured facts carry the diagnosis, and both are readable from the defect
log alone, without the control arm:

1. **`critic/advantages/min` is exactly `0.0` on all 60 steps.** GRPO cannot
   produce an all-non-negative advantage batch when groups mix correct and
   incorrect responses; the control arm goes negative on 29 of 60 steps.
2. **`critic/advantages/max` is bit-identical on every rewarded step:**
   `0.9999990463256836`, which is fp32 `1/(1+1e-6)` — raw reward `1.0`
   divided by the substituted `id2std = 1.0` plus the `epsilon = 1e-6` guard.
   The control arm shows three different maxima, one per group composition.

Two corrections to what packaging predicted before the run, both now fixed in
`oracle.md`:

- `critic/advantages/mean` does **not** equal `critic/rewards/mean` exactly.
  It is a response-token-weighted masked mean over response tokens, so it
  tracks the reward mean only approximately (e.g. step 4: adv `0.010730` vs
  reward `0.015625`). Testing for exact equality would reject a correct
  diagnosis. The sign and the absence of cancellation are the real signal.
- The held-out curve does **not** collapse. At a 60-step budget the defect arm
  ends slightly *higher* than it started (0.0454 → 0.0570) and is within noise
  of the control arm (0.0592). A diagnosis resting on held-out accuracy has not
  found the defect — which makes this a good discriminating task, but it does
  mean a solution-level threshold cannot use this metric at this budget.

The pre-run CPU derivation did get the control arm's `2.4748666286` exactly
right (`(1 − 1/8)/0.35355`), which is why the two maxima columns above agree
with the sibling task's independently frozen log.

## Evaluation design

Issue level only. The shared LLM judge compares
`$PTDBENCH_RESULTS/agent_answer.md` against the hidden `oracle.md` on verdict
/ root cause / evidence with weights 0.2 / 0.6 / 0.2 and a pass bar of 0.8.
There is a single root cause, so `oracle.md` carries no partial-credit clause.
It does carry the opposite requirement: "group size is 1" or "advantages are
not normalised", stated without locating the uid assignment relative to
`batch.repeat`, is incomplete and must not get full `root_cause_score`. The
symptom is one grep away in the log; the cause is one indirection further, in
the silent singleton-group branch.

`oracle.md` also lists the wrong attributions to reject: `train_batch_size`,
`rollout.n`, the learning rate, the reward function, the absence of KL loss,
and a bare "held-out accuracy is low". All of those match the healthy recipe.

### Why issue-only

Both arms have now been run, so the blocker is no longer missing data — it is
that **this recipe has no metric that separates the arms at a 60-step budget.**
Held-out accuracy ends 0.0570 (defect) vs 0.0592 (control), well inside
sampling noise, and training reward is 0.0161 vs 0.0156. A solution-level
grader reading `$PTDBENCH_RESULTS/experiment_logs` the way every other task's
does would be grading noise.

The quantities that *do* separate the arms cleanly — `critic/advantages/min`
being identically zero, and `critic/advantages/max` pinned to
`0.9999990463256836` — are structural invariants, not scores. A solution level
for this task should grade those directly from the new run's metric JSONL
rather than invent an accuracy threshold:

- pass requires `min(critic/advantages/min) < 0` over the graded steps, and
- more than one distinct value of `critic/advantages/max`, and
- every pinned step present (so a 3-step run cannot pass by having no
  rewarded steps at all, where the defect's `max` is legitimately `0.0`).

That is a different grader shape from the repo's existing
`grade_submission.py`, which is why it is not bolted on here. The alternative
— extending the budget until accuracy separates — was not measured and should
not be assumed; at 2 GPUs, 60 steps is 13.5 min, so a 400-step calibration
pair is about 3 GPU-hours if someone wants to try.

### Reproducing the two arms

Defect arm, from `workspace/`:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash examples/grpo_trainer/run_llama3_2_1b_instruct_dapo_math_fsdp.sh
```

Control arm: in a *separate* copy of the workspace, move the four-line
`batch.non_tensor_batch["uid"] = ...` block in
`verl/trainer/ppo/ray_trainer.py` back above
`gen_batch = self._get_gen_batch(batch)`, then run the identical command. No
launch-script or config change is involved, so the two logs differ only by
the defect. Verify with
`sha256sum verl/trainer/ppo/ray_trainer.py` — the repaired file must be
`e636912ae2be46b6ff7152f6473feab3af5db962f744903dc55f7d26d58d7d73`, the
pristine upstream blob. Keep the control arm's code and logs out of
`workspace/`.

Per-step metrics land in `logs/metrics_<timestamp>.jsonl`, which is the file
to read: the launch script enables verl's `file` tracking backend via
`VERL_FILE_LOGGER_PATH` precisely because the `console` backend's metric lines
are printed inside the TaskRunner Ray actor and **Ray's forwarding of that
actor's stdout into the launch log is not dependable** — the first recording
attempt on this very image produced a log file with zero `step:N` lines while
the Ray worker log had all 61. The `file` backend writes unbuffered and does
not depend on Ray at all.

Check step 1 before letting either run finish: `critic/advantages/min` decides
the whole experiment and is visible immediately.

## Environment requirement

The task runs on the `verl` PTDBench image (`environment: image: verl`, built
from `docker/images/verl.Dockerfile`) — verl + vLLM 0.17, torch 2.10 /
CUDA 12.9 on Ubuntu 22.04. It is the image the sibling task's recorded run
used with this same verl commit, which is why the staged tree is known to
import and launch there. The CPU-only `rank-distillation-cpu` image cannot
host a vLLM rollout, and the slime images are a different framework.

`environment/prepare.sh` delivers the Llama-3.2-1B-Instruct weights to
`workspace/verl/Llama-3.2-1B-instruct` (note the lowercase `i`, which is what
the launch script looks for) and the DAPO-Math-17k train/test parquet pair to
`workspace/DAPO-Math-17k_verl`, both pinned by commit sha and checksummed.
Its non-comment lines are byte-identical to the sibling task's `prepare.sh`,
so the two tasks share one dataset cache entry.

The recipe needs **2 GPUs**; `metadata.yaml` declares them through
`environment.env`.
