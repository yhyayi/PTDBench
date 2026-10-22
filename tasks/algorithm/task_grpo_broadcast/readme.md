# reward_design implementation diagnosis: task_grpo_broadcast

## Case overview

- Evaluation key: `reward_design/task_grpo_broadcast`
- Case family: `reward_design`
- Schema version: `2`
- Evaluation levels: `issue`, `solution`
- Model: `Qwen2.5-1.5B-Instruct`
- Training framework: `verl`
- Primary source evaluation metric: `val-core/taco/acc/mean@1`

This document is for maintainers and reviewers. It is not exposed to the
agent during evaluation. Metric names preserve the source experiment;
tasks using the same held-out protocol may intentionally share a name.

## Canonical failure reason

Two defects:

- `compute_grpo_outcome_advantage` returns shape `[batch, response_length]`, but the per-sample advantages are written along the sequence axis and broadcast down every row before the response mask.
- `taco.py::_to_bool_results` calls `e.item(0)` whenever a test result is a NumPy array. A call-based completion that returns an empty array makes `output == expected` a size-0 array, and `.item(0)` raises `IndexError`, which aborts the training job. The step is whichever rollout first returns that value.

## Workspace and hidden evaluations


| Path                    | Visibility and purpose                                                                     |
| ----------------------- | ------------------------------------------------------------------------------------------ |
| `workspace/`            | Agent-visible experiment snapshot. Historical logs and wandb were not copied.             |
| `evaluations/issue/`    | Neutral diagnosis instruction, issue oracle, and task-local issue verifier.                |
| `evaluations/solution/` | Repair-and-run instruction, hidden reference material, and metric-only verifier.           |


The agent cannot read oracle or verifier paths. `issue` grading compares
`$PTDBENCH_RESULTS/agent_answer.md` to hidden `oracle.md` with an LLM
judge. `solution` grading reads only the new experiment logs.

## Source and preserved context

```yaml
bench_category: reward_design
source_task: data_call/tasks/reward_design/task_function_call
evidence_kind: injected_implementation_defect
packaging_tier: B0
packaging_ready: true
b1_scoring_ready: true
target_files:
  - verl/trainer/ppo/core_algos.py
  - verl/utils/reward_score/taco.py
validation_stack: TACO_EASY
already_at_baseline: false
metric_status: not_run
verify:
  num_gpus: 8
  timeout_minutes: 660
  max_train_steps: 400
  eval_metric_key: val-core/taco/acc/mean@1
```

The copy started from `task_function_call`. That package's workspace `taco.py`
still forces stdin scoring. This task restores the `fn_name` call-based path
from that package's healthy scorer, then keeps a second defect already present
in that scorer: `_to_bool_results` uses `e.item(0)` on array results.
`task_function_call` itself is unchanged. Logs, wandb, `agent_trace.json`,
and `workspace/outputs` were not copied, so the function-call curve is not
evidence for these defects.

## Original failure evidence

No fresh 400-step curve is stored for these defects. Solution grading
still requires a new run and reads only that run's experiment logs.

## Canonical expected outcome

Verdict: **unhealthy**.

Root cause targets, both required:

- `verl/trainer/ppo/core_algos.py::compute_grpo_outcome_advantage`
- `verl/utils/reward_score/taco.py::_to_bool_results`

Required fix targets:

- `verl/trainer/ppo/core_algos.py::compute_grpo_outcome_advantage`: `restore`; target state/value: `scores = scores.unsqueeze(-1) * response_mask`, so each row carries that sample's scalar GRPO advantage on its response tokens.
- `verl/utils/reward_score/taco.py::_to_bool_results`: stop calling `.item(0)` on a possibly empty `ndarray`. A size-0 comparison result must become a failed case and must not escape `compute_score`.

## Evaluation design

`issue` grades `$PTDBENCH_RESULTS/agent_answer.md` against hidden
`oracle.md`. The judge scores verdict, root-cause, and evidence agreement
(weights 0.2 / 0.6 / 0.2). Pass is a weighted score of at least `0.9`.
`root_cause_score` is `1` only when both causes above are identified. If
either cause is missing, `root_cause_score` is at most `0.5`. With the other
two scores at `1`, that caps the weighted score at `0.7`, so a one-cause
diagnosis does not pass.

`solution` does not grade a structured diagnosis. The agent must repair the
staged workspace, run the designated launch script for 400 steps, and produce
fresh experiment logs. Scoring is weighted:

- validation: a finite `val-core/taco/acc/mean@1` peak of at least `0.125`
  (12.5%) scores `1`, otherwise `0`
- training reward: the mean `critic/rewards/mean` over training steps must be
  strictly above `0.14` to score `1`, otherwise `0`
- final `score = 0.8 * validation + 0.2 * training reward`

Pass requires both components (`score == 1.0`). The framework still records
the weighted float when only one component passes.

## External dataset asset

- Repository: `LIF1014/ptdbench-reward-design-task-grpo-broadcast-dataset`
- Publication status: `published`
- Pinned revision: `bf47407d7372856d5d09b6d16dae3c1570393692`
- Dataset license: `Apache-2.0`
- Provenance: Same TACO EASY snapshot as `task_function_call` (8368 train / 184 test).


| Runtime path                            | Size              | SHA-256                                                            |
| --------------------------------------- | ----------------- | ------------------------------------------------------------------ |
| `workspace/taco_verl/test.parquet`      | 61,082,676 bytes  | `e364fa69d973f8e8644b2fa4d7dc6ec405d5b4adff64002c6ad52e7c787c6eaa` |
| `workspace/taco_verl/train.parquet`     | 606,016,648 bytes | `7f25abdfa9e0f5baa173d153a32973fbe2cc8736503a35408b8ebb8ae2bc5706` |


`environment/prepare.sh` downloads the immutable revision in the prepare
stage, verifies the archive and the unpacked parquet bytes, and stages
`workspace/taco_verl` before the agent runs.

## External model asset

- Repository: `Qwen/Qwen2.5-1.5B-Instruct` (official)
- Pinned revision: `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- Byte-identical, on every allowed file, to the former task snapshot
  `LIF1014/ptdbench-verl-implementation-torch-functional@36a4c22`; switched
  because that repo name is visible to the agent through the shared download cache.
- Model: `Qwen2.5-1.5B-Instruct` (`model.safetensors` SHA-256 `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`)

## Environment requirement

- Base image: `verlai/verl:vllm017.latest`
- Use the Docker agent backend after external asset publication.
- The local backend does not build the task image or hydrate external assets.
- The designated experiment uses 8 GPUs, a 400-step budget, and a large shared-memory allocation.
- Launch script: `examples/grpo_trainer/run_qwen2_5_1.5b_instruct_taco_fsdp.sh`

