# verl_reward_implementation implementation diagnosis: task_engine_base_025

## Case overview

- Evaluation key: `verl_implementation/verl_reward_implementation/task_engine_base_025`
- Case family: `verl_reward_implementation`
- Schema version: `2`
- Evaluation levels: `issue`, `solution`
- Model: `Qwen2.5-0.5B-Instruct`
- Training framework: `verl`
- Primary source evaluation metric: `val-core/openai/gsm8k/acc/mean@1`

This document is for maintainers and reviewers. It is not exposed to the
agent during evaluation. Metric names preserve the source experiment;
tasks using the same held-out protocol may intentionally share a name.

## Canonical failure reason

Rewritten BaseEngine, BaseEngineCtx, and EngineRegistry methods collapse GSM8K validation accuracy to zero with no reward rise versus a 47.8% baseline, while the `####`-only GSM8K scorer does not accept Qwen2.5-0.5B-Instruct's preferred `\boxed{N}` format

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
bench_category: verl_reward_implementation
source_task: Benchmark_Evaluation/tasks/task_verl_reward_implementation/task_engine_base_025
evidence_kind: dirty_agent_failure
packaging_tier: B0
packaging_ready: true
b1_scoring_ready: false
target_files: verl/workers/engine/base.py
validation_stack: GSM8K_G3
already_at_baseline: false
metric_status: log_reconstructed
verify:
  num_gpus: 2
  timeout_minutes: 90
  max_train_steps: 100
  eval_metric_key: val-core/openai/gsm8k/acc/mean@1
historical_baseline:
  actual_metric_key: val-core/openai/gsm8k/acc/mean@1
  eval_last: 0.0
  eval_max: 0.001516300227445034
  n_eval_points: 6
  attempt: 1
```

## Original failure evidence

- val curve 0.0008 → 0.0; baseline under same setup reaches 47.8%
- no reward rise within 100 steps
- pinned GSM8K prompts request a final `####` answer and the default scorer honors that nominal contract, but Qwen2.5-0.5B-Instruct preferentially emits `\boxed{N}`, so almost every rollout is scored 0

The preserved source metric is `val-core/openai/gsm8k/acc/mean@1`. The frozen workspace remains
the evidence package; a full training rerun is outside verifier scope.

## Canonical expected outcome

Verdict: **unhealthy**.

Root cause targets:

- `verl/workers/engine/base.py`
- `verl/utils/reward_score/gsm8k.py::compute_score`

Required fix targets:

- `verl/workers/engine/base.py`: `restore`; target state/value: baseline BaseEngine, BaseEngineCtx, and EngineRegistry semantics.
- `verl/utils/reward_score/gsm8k.py::compute_score`: `align_format_contract`; target state/value: accept `#### N` and fall back to boxed / last numeric answer.

Documentation reconciliation:

- Pinned parquet inspection shows a nominal prompt/extractor match: train and test prompts ask to output the final answer after `####`, and `compute_score` defaults to `method="strict"`.
- The format defect is a data-format / model-convention gap: this Qwen checkpoint prefers `\boxed{N}` (and other non-`####` numeric endings), so the `####`-only contract yields sparse or zero GRPO signal and held-out strict accuracy collapses.
- Either listed root cause is a correct attribution.

## Evaluation design

`issue` grades `$PTDBENCH_RESULTS/agent_answer.md` against hidden
`oracle.md`. The judge scores verdict, root-cause, and evidence agreement
(weights 0.2 / 0.6 / 0.2). Pass is a weighted score of at least `0.8`.
Either listed root cause is independently sufficient; naming one with the
correct causal role should receive full `root_cause_score`.

`solution` does not grade a structured diagnosis. The agent must repair the
staged workspace, run the designated launch script, and produce a finite
`val-core/openai/gsm8k/acc/mean@1` at step 100 of at least `0.4` (40%).
Pass is score `1`; otherwise `0`.

## External dataset asset

- Repository: `LIF1014/ptdbench-verl-reward-implementation-task-engine-base-025-dataset`
- Publication status: `published`
- Pinned revision: `ec9d2e88d7729dd73c81715750e8a34ad9718c49`
- Dataset license: `MIT`
- Provenance: Processed runtime snapshot of openai/gsm8k.


| Runtime path                         | Size            | SHA-256                                                            |
| ------------------------------------ | --------------- | ------------------------------------------------------------------ |
| `workspace/gsm8k_verl/test.parquet`  | 1,183,034 bytes | `0dcd50ed32caa9d8f434d4e3ef8031bc3e49b0f823655a53be5c002e727f17dd` |
| `workspace/gsm8k_verl/train.parquet` | 6,555,602 bytes | `8fb13a0cd8621e5fcacdee3ffe28b3a369975478ea7ab514781f3f190cba2488` |


The Docker image downloads the immutable revision at build time, verifies
the artifact, and hydrates the bind-mounted workspace only for the agent stage.

## Environment requirement

- Base image: `verlai/verl:vllm017.latest`
- Use the Docker agent backend after external asset publication.
- The local backend does not build the task image or hydrate external assets.
- Model weights and historical training checkpoints are not dataset assets.

