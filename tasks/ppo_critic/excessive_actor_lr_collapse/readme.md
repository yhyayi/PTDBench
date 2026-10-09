# PPO-Critic run assessment: excessive_actor_lr_collapse

## Case overview

- Evaluation key: `excessive_actor_lr_collapse`
- Case family: `ppo_critic`
- Schema version: `2`
- Evaluation levels: `issue`, `solution`
- Model: `Qwen3-4B-Base`
- Dataset: `GuruMath controlled smoke` (RL prompts) + AIME-32 / MATH-500-32 held-out
- Training framework: `verl` (VIMPO PPO-Critic recipe)
- Launch script: `run_buggy.sh` → `recipe/vimpo/run_vimpo_critic.sh`
- Primary source evaluation metric: `val-core/math__math/acc/mean@1`

This document is for maintainers and reviewers. It is not exposed to the agent
during evaluation.

## Canonical verdict

**unhealthy.**

The staged workspace reproduces the VIMPO PPO-Critic recipe on an unmodified
verl snapshot, but the actor learning rate has been inflated. `run_buggy.sh`
exports `LR=1e-2`; the recipe `recipe/vimpo/run_vimpo_critic.sh` passes it as
`actor_rollout_ref.actor.optim.lr="${LR:-1e-6}"`, so the actor optimizer runs at
`1e-2` — 10,000x the intended `1e-6`. The recipe file itself is byte-for-byte
identical to the healthy control; the critic LR is unchanged (`CRITIC_LR=1e-5`),
confirming only the actor LR was tampered with.

## Workspace and hidden evaluations

| Path | Visibility and purpose |
| --- | --- |
| `workspace/` | Agent-visible historical experiment snapshot; evidence rather than canonical ground truth. |
| `workspace/ptdbench_case/` | Preserved controlled evidence: assets and `logs/observed_buggy_run.log`. |
| `evaluations/issue/` | Neutral diagnosis instruction, issue oracle, and task-local LLM-judge verifier. |
| `evaluations/solution/` | Repair-and-run instruction, hidden reference material, and metric-only verifier. |

The agent cannot read oracle or verifier paths. `issue` grading compares
`$PTDBENCH_RESULTS/agent_answer.md` to hidden `oracle.md` with an LLM judge.
`solution` grading reconstructs `val-core/math__math/acc/mean@1` and
`critic/rewards/mean` from the logs the agent's own rerun writes under
`$PTDBENCH_RESULTS/experiment_logs`.

## Original failure evidence

From `workspace/ptdbench_case/logs/observed_buggy_run.log` (8 PPO steps,
`test_freq=1`):

- the job initializes and completes all 8 steps (not a startup/infra failure)
- `critic/rewards/mean` starts non-zero (≈0.03–0.06) then falls to `0.0`
- validation starts non-zero and reaches zero on both `val-core/math__math` and
  `val-core/math__aime_repeated_8x`
- responses saturate the 1024-token `MAX_RESPONSE_LENGTH`
- `actor/grad_norm` and actor entropy become highly unstable

This is the classic signature of policy collapse driven by an over-aggressive
actor learning rate.

## Why the recipe is unhealthy

Apart from the actor LR, the recipe is reproduced correctly. The verl framework
code under `workspace/verl/` and `workspace/recipe/` is byte-for-byte identical
to the healthy control; only `run_buggy.sh` differs (`LR=1e-2`).

- **Hyperparameter defect** (`run_buggy.sh`, `export LR=1e-2`): consumed by
  `recipe/vimpo/run_vimpo_critic.sh` as
  `actor_rollout_ref.actor.optim.lr="${LR:-1e-6}"`. The actor optimizer therefore
  takes gradient steps 10,000x too large and the policy degenerates.

The surrounding recipe is otherwise correct — these are not the defect: the
critic LR (`1e-5`), the FSDP/vLLM wiring, and the controlled smoke downscaling
(4 GPUs, 8 steps, batch 16, 1024-token responses) are expected harness
properties, not causes.

## Canonical expected outcome

Verdict: **unhealthy**.

- Root cause target: the inflated actor learning rate `LR=1e-2` in
  `run_buggy.sh`.
- Required fix target: resolve the actor optimizer LR to `1e-6` on the real
  launcher-to-optimizer path — either set `export LR=1e-6` in `run_buggy.sh` or
  remove the override and let the recipe default (`${LR:-1e-6}`) apply.

An agent that attributes the collapse to an infrastructure-startup failure, the
critic LR, or an unusual-but-benign setting is wrong.

## Evaluation design

`issue` grades `$PTDBENCH_RESULTS/agent_answer.md` against hidden `oracle.md`.
The judge scores verdict, root-cause, and evidence agreement (weights
0.2 / 0.6 / 0.2). Pass is a weighted score of at least `0.8`.

`solution` is deterministic and metric-only, in the same shape as
`verl_implementation/torch_functional`: fix the LR, rerun the launch script for
real, and the verifier reconstructs the metric from the logs under
`$PTDBENCH_RESULTS/experiment_logs`. Score is
`0.8 * validation + 0.2 * training_reward`; pass requires both components
(`score == 1.0`). See `evaluations/solution/verifier/resources/README.md`.

> **Threshold calibration (maintainers).** The historical baseline in
> `grader_spec.json` is the buggy run's real terminal value (`0.0`). The
> `required_final_score` / `required_training_reward` are evidence-based
> conservative drafts and are **not yet calibrated** against a real healthy
> rerun; `task.yaml` carries `needs_manual_review: true`. Calibrate against an
> oracle-fixed run (`LR=1e-6`) before using this task for scoring.

## Environment

`metadata.environment.image: ppo-critic-vimpo` resolves to the shared image
`docker/images/ppo-critic-vimpo.Dockerfile` (Python 3.12, CUDA 12.8, torch
2.8.0, vLLM 0.11.0, Ray 2.56.0, transformers 4.57.6, flash-attn 2.8.3.post1).
This stack is deliberately **not** the shared `verl` image (vLLM 0.17 / torch
2.10 / CUDA 12.9): the workspace ships its own veRL + VIMPO recipe written
against vLLM 0.11, and the recorded val-core / critic-reward baselines the
solution grader reconstructs were produced on this exact stack. The workspace
veRL is not baked into the image; the harness bind-mounts the workspace at
`/workspace` and `PYTHONPATH=/workspace` makes `import verl` resolve to it.
`environment/prepare.sh` fetches the Qwen3-4B-Base base model to
`/workspace/Qwen3-4B-Base` (the `MODEL_PATH` the effective run uses, per
`solution/instruction.md`). Its `--revision` is **pinned** to
canonical `Qwen/Qwen3-4B-Base` at main as of 2026-08-18 — commit
`906bfd4b4dc7f14ee4320094d8b41684abff8539`, the snapshot the recorded runs used
(verified by downloading the full repo locally and checking each shard) — and
the three weight shards additionally carry `--expect` sha256 + byte-size guards.
`QWEN3_4B_BASE_REVISION` and `QWEN3_4B_BASE_REPO` remain overridable (e.g. to a
staged mirror if canonical Qwen's xet-only hosting is blocked on-cluster;
override both together, since a mirror's commit differs). The controlled
training and held-out assets are tiny and ship embedded in
`workspace/ptdbench_case/assets/*.parquet`, so prepare does not re-fetch them;
the dependency stack is baked into the shared image and the workspace veRL needs
no editable install (`import verl` resolves via `PYTHONPATH=/workspace`;
`__version__` comes from `verl/version/version`).

The recipe needs **4 GPUs** (`run_buggy.sh`: `CUDA_VISIBLE_DEVICES=0,1,2,3`,
`N_GPUS_PER_NODE=4`); `MODEL_PATH` must point at a local Qwen3-4B-Base checkpoint
(a large external resource, not embedded).

The workspace code and the verl snapshot are distributed under Apache-2.0; the
task packaging is MIT.
