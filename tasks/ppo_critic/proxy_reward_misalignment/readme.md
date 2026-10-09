# PPO-Critic run assessment: proxy_reward_misalignment

## Case overview

- Evaluation key: `proxy_reward_misalignment`
- Case family: `ppo_critic`
- Schema version: `2`
- Evaluation levels: `issue`, `solution`
- Model: `Qwen3-4B-Base`
- Dataset: `GuruMath fixed-16 controlled set` (RL prompts) + AIME-32 / MATH-500-32 held-out
- Training framework: `verl` (VIMPO PPO-Critic recipe)
- Launch script: `run_buggy.sh` → `recipe/vimpo/run_vimpo_critic.sh`
- Primary source evaluation metric: `val-core/math__math/acc/mean@1`

This document is for maintainers and reviewers. It is not exposed to the agent
during evaluation.

## Canonical verdict

**unhealthy.**

The staged workspace reproduces the VIMPO PPO-Critic recipe on an unmodified
verl snapshot, but a misspecified **proxy reward** has been injected through a
custom reward function. `run_buggy.sh` wires
`reward.custom_reward_function.path=ptdbench_case/type3_length_reward_v2_fn.py`.
For the controlled training source (`data_source == "ptdbench_type3_proxy_train_v2"`)
that function returns a **brevity proxy** as the training score:

```text
proxy_reward = max(0.0, min(1.0, 1.0 - n_chars / 4000.0))
```

Shorter responses receive higher reward regardless of mathematical correctness.
Validation (`data_source != TRAIN_SOURCE`) still scores genuine correctness, so
the training objective is misaligned with the target task. This is a
reward-hacking / proxy-objective-misalignment pattern, not a dead reward signal.

## Workspace and hidden evaluations

| Path | Visibility and purpose |
| --- | --- |
| `workspace/` | Agent-visible historical experiment snapshot; evidence rather than canonical ground truth. |
| `workspace/ptdbench_case/` | Preserved controlled evidence: injected `type3_length_reward_v2_fn.py`, controlled assets, and `logs/observed_buggy_run.log`. |
| `evaluations/issue/` | Neutral diagnosis instruction, issue oracle, and task-local LLM-judge verifier. |
| `evaluations/solution/` | Repair-and-run instruction, hidden reference material, and metric-only verifier. |

The agent cannot read oracle or verifier paths. `issue` grading compares
`$PTDBENCH_RESULTS/agent_answer.md` to hidden `oracle.md` with an LLM judge.
`solution` grading reconstructs `val-core/math__math/acc/mean@1` and
`critic/rewards/mean` from the logs the agent's own rerun writes under
`$PTDBENCH_RESULTS/experiment_logs`.

## Original failure evidence

From `workspace/ptdbench_case/logs/observed_buggy_run.log` (15 PPO steps,
`test_freq=5`, `val_before_train=True`):

- `critic/rewards/mean` (the proxy) is **high and rising**: `~0.52 → 0.67`
  across steps, mean `~0.54`.
- Held-out `val-core/math__math/acc/mean@1` **declines**: `0.78125 (step 0)` →
  `0.6875 (step 5)` → `0.65625 (step 10)` → `0.59375 (step 15)`.
- `response_length/mean` falls (~`677 → 474`): the policy is learning to be
  shorter, exactly what the proxy rewards.
- Auxiliary truth tracking (`true_accuracy/score_mean`) is ~`0.52` and does not
  improve, confirming the proxy gain does not reflect correctness.

PPO initializes and completes all 15 steps; gradients, critic updates, and MFU
look healthy. The defect is in the *objective*, not the optimizer.

## Why the recipe is unhealthy

The verl framework code under `workspace/verl/` and `workspace/recipe/` is
byte-for-byte identical to the healthy control; only `run_buggy.sh` and the
injected `ptdbench_case/type3_length_reward_v2_fn.py` differ.

- **Reward defect** (`ptdbench_case/type3_length_reward_v2_fn.py::compute_score`):
  for the training source the returned `score` is `1 - n_chars/4000` (brevity),
  not correctness. The function even carries truthful `acc` /
  `true_math_score` fields, so the misalignment is observable: reward goes up
  while `true_math_score` and held-out accuracy do not.

The surrounding recipe is otherwise correct — these are not the defect: the
VIMPO PPO-Critic algorithm/optimizer config (actor `LR=5e-6`, critic
`CRITIC_LR=1e-5`), the FSDP/vLLM wiring, and the controlled smoke downscaling
(4 GPUs, 15 steps, fixed-16 train set repeated x64, 1024-token responses) are
expected harness properties, not causes.

## Canonical expected outcome

Verdict: **unhealthy**.

- Root cause target: the brevity proxy in
  `ptdbench_case/type3_length_reward_v2_fn.py::compute_score` for the training
  source.
- Required fix target: score the training source by mathematical correctness
  (a short incorrect answer must score 0; both short and long correct boxed
  answers must score 1; response length must not determine the training
  reward), keeping the `compute_score(data_source, solution_str, ground_truth,
  extra_info=None, **kwargs)` signature. `oracle/fixed_reward_fn.py` routes the
  training source through `default_compute_score` on `math__math`.

An agent that attributes the symptom to a dead reward, a parser mismatch, an
actor-LR collapse, or a startup failure is wrong.

## Evaluation design

`issue` grades `$PTDBENCH_RESULTS/agent_answer.md` against hidden `oracle.md`.
The judge scores verdict, root-cause, and evidence agreement (weights
0.2 / 0.6 / 0.2). Pass is a weighted score of at least `0.8`.

`solution` is deterministic and metric-only, in the same shape as
`verl_implementation/torch_functional`: fix the reward, rerun the launch script
for real, and the verifier reconstructs the metric from the logs under
`$PTDBENCH_RESULTS/experiment_logs`. Score is
`0.8 * validation + 0.2 * training_reward`; pass requires both components
(`score == 1.0`). See `evaluations/solution/verifier/resources/README.md`.

> **Reward-hacking subtlety (maintainers — read before scoring).** This task
> needs non-default grader handling and is the least self-calibrating of the
> four:
>
> - Validation is graded on the **terminal** (final-step) `val-core/math__math/acc/mean@1`,
>   **not a peak**. The buggy run's accuracy *peaks at step 0* (`0.78125`, before
>   training corrupts it) and then declines to `0.59375`. A peak selector would
>   let an unfixed rerun pass, so the verifier (`grade_submission.py::select_final`)
>   deliberately uses the value at the latest step ≤ `final_step`.
> - `critic/rewards/mean` is **high by construction** in the buggy run (it is the
>   proxy, ~`0.54`). The `training_reward` component therefore does **not**
>   discriminate fixed from unfixed for this task; the terminal validation
>   accuracy is the real discriminator. `required_training_reward` stays a low
>   floor (`0.05`) only to catch a catastrophically broken rerun.
> - `historical_final_baseline` is the buggy terminal (`0.59375`).
>   `required_final_score = 0.65` is an **uncalibrated conservative draft** placed
>   just above the hacked terminal and below the step-0 baseline; it demands the
>   fix stop the decline. It has **not** been verified against a real
>   correctness-fixed rerun and may need adjustment. `task.yaml` carries
>   `needs_manual_review: true`.

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
training set (`train_proxy_fixed16_x64.parquet`) and held-out assets
(`aime32.parquet`, `math500_32.parquet`) are tiny and ship embedded in
`workspace/ptdbench_case/assets/`, so prepare does not re-fetch them; the
dependency stack is baked into the shared image and the workspace veRL needs no
editable install (`import verl` resolves via `PYTHONPATH=/workspace`;
`__version__` comes from `verl/version/version`).

The recipe needs **4 GPUs** (`run_buggy.sh`: `CUDA_VISIBLE_DEVICES=0,1,2,3`,
`N_GPUS_PER_NODE=4`); `MODEL_PATH` must point at a local Qwen3-4B-Base checkpoint
(a large external resource, not embedded).

The workspace code and the verl snapshot are distributed under Apache-2.0; the
task packaging is MIT.
