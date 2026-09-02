# verl_coding implementation diagnosis: task_function_call

## Case overview

- Evaluation key: `verl_coding/task_function_call`
- Case family: `verl_coding`
- Schema version: `2`
- Evaluation levels: `issue`, `solution`
- Model: `Qwen2.5-1.5B-Instruct`
- Training framework: `verl`
- Primary source evaluation metric: `val-core/taco/acc/mean@1`

This document is for maintainers and reviewers. It is not exposed to the
agent during evaluation. Metric names preserve the source experiment;
tasks using the same held-out protocol may intentionally share a name.

## Canonical failure reason

The staged `taco.py` ignores `fn_name` and forces every sample through the stdin path, so function-call problems always receive zero reward while stdin problems continue to score normally.

## Workspace and hidden evaluations


| Path                    | Visibility and purpose                                                                     |
| ----------------------- | ------------------------------------------------------------------------------------------ |
| `workspace/`            | Agent-visible historical experiment snapshot; evidence rather than canonical ground truth. |
| `evaluations/issue/`    | Neutral diagnosis instruction, issue oracle, and task-local issue verifier.                |
| `evaluations/solution/` | Repair-and-run instruction, hidden reference material, and metric-only verifier.           |


The agent cannot read oracle or verifier paths. `issue` grading compares
`$PTDBENCH_RESULTS/agent_answer.md` to hidden `oracle.md` with an LLM
judge. `solution` grading reads only the new experiment logs.

## Source and preserved context

The following source and historical packaging fields were moved from
`metadata.yaml` so the machine metadata remains aligned with the core
schema. They are preserved here without changing their meaning.

```yaml
bench_category: verl_coding
source_task: Benchmark_Evaluation/tasks/task_verl_coding/tasks_function_call
evidence_kind: injected_implementation_defect
packaging_tier: B0
packaging_ready: true
b1_scoring_ready: true
target_files: verl/utils/reward_score/taco.py
validation_stack: TACO_EASY
already_at_baseline: false
metric_status: log_reconstructed
verify:
  num_gpus: 8
  timeout_minutes: 960
  max_train_steps: 400
  eval_metric_key: val-core/taco/acc/mean@1
historical_baseline:
  actual_metric_key: val-core/taco/acc/mean@1
  eval_last: 0.05434782608695652
  eval_max: 0.08695652173913043
  n_eval_points: 9
  attempt: 1
```

## Original failure evidence

- dirty validation rises from 0.0217 to 0.0543 and peaks at 0.0870 at step 300; the healthy scorer under the same setup reaches 0.0598 at step 0 and 0.139 at step 400
- official function solutions score 0 on both train and validation, while official stdin solutions still score 1
- the held-out set contains 41 call-based and 143 stdin problems, so every call-based row is capped at zero even for a correct function solution

The preserved source metric is `val-core/taco/acc/mean@1`. The frozen workspace remains
the evidence package; solution grading requires a new 400-step training run and reads only
its experiment logs.

## Canonical expected outcome

Verdict: **unhealthy**.

Root cause targets:

- `verl/utils/reward_score/taco.py::run_test`

Required fix targets:

- `verl/utils/reward_score/taco.py::run_test`: `restore`; target state/value: when `fn_name` is present, compile the candidate as call-based code, instantiate `Solution` when required, invoke `method(*inputs)`, and compare its return value; leave the stdin path unchanged.

Documentation reconciliation:

- The parquet data and prompts still identify call-based problems through `fn_name` and `Use Call-Based format`; the defect is confined to the scorer.
- The retained stdin path explains why aggregate validation remains nonzero despite all call-based rows failing.
- `prime_code` and agent-loop tool parsers are unused by the designated TACO launch and are not causal.

## Evaluation design

`issue` grades `$PTDBENCH_RESULTS/agent_answer.md` against hidden
`oracle.md`. The judge scores verdict, root-cause, and evidence agreement
(weights 0.2 / 0.6 / 0.2). Pass is a weighted score of at least `0.8`.
Ignoring `fn_name` and deleting the call-based execution path is the essential
root cause.

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

- Repository: `unpublished`
- Publication status: `unpublished; stored in the prebuilt task image`
- Pinned revision: `not applicable`
- Dataset license: `not recorded in the task package`
- Provenance: Processed TACO training and held-out snapshot.


| Runtime path                            | Size              | SHA-256                                                            |
| --------------------------------------- | ----------------- | ------------------------------------------------------------------ |
| `workspace/taco_verl/test.parquet`      | 61,082,676 bytes  | `e364fa69d973f8e8644b2fa4d7dc6ec405d5b4adff64002c6ad52e7c787c6eaa` |
| `workspace/taco_verl/train.parquet`     | 606,016,648 bytes | `7f25abdfa9e0f5baa173d153a32973fbe2cc8736503a35408b8ebb8ae2bc5706` |


The prebuilt Docker image embeds the dataset snapshot and hydrates the
bind-mounted workspace only for the agent stage. The current Dockerfile does
not download or verify a published external dataset revision.

## Environment requirement

- Base image: `verlai/verl:vllm017.latest`
- Use the Docker agent backend with the task-specific prebuilt image.
- The local backend does not hydrate the baked model and dataset assets.
- The designated experiment uses 8 GPUs, a 400-step budget, and a large shared-memory allocation.
- Model weights and historical training checkpoints are not external dataset assets.
