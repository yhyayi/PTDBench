# llama_dapo_implementation implementation diagnosis: task_monkey_patch_011

## Case overview

- Evaluation key: `verl_implementation/llama_dapo_implementation/task_monkey_patch_011`
- Case family: `llama_dapo_implementation`
- Schema version: `2`
- Evaluation levels: `issue`, `solution`
- Model: `Llama-3.2-1B-Instruct`
- Dataset: `DAPO-Math-17k`
- Training framework: `verl`
- Primary source evaluation metric: `val-core/math_dapo/acc/mean@1`

This document is for maintainers and reviewers. It is not exposed to the
agent during evaluation. Metric names preserve the source experiment;
tasks using the same held-out protocol may intentionally share a name.

## Canonical failure reason

A rewritten apply_monkey_patch no longer installs the baseline flash-attention hook, cutting val-core 43% below control with 70.5% zero-advantage steps, while `TRAIN_BATCH_SIZE=4` with sparse ~3–5% 0/1 rewards yields mostly all-incorrect GRPO groups and zero advantage

## Workspace and hidden evaluations

| Path | Visibility and purpose |
| --- | --- |
| `workspace/` | Agent-visible historical experiment snapshot; evidence rather than canonical ground truth. |
| `evaluations/issue/` | Neutral diagnosis instruction, issue oracle, and task-local issue verifier. |
| `evaluations/solution/` | Repair-and-run instruction, hidden reference material, and metric-only verifier. |

The agent cannot read oracle or verifier paths. `issue` grading compares
`$PTDBENCH_RESULTS/agent_answer.md` to hidden `oracle.md` with an LLM
judge. `solution` grading reads only the new experiment logs.

## Source and preserved context

The following source and historical packaging fields were moved from
`metadata.yaml` so the machine metadata remains aligned with the core
schema. They are preserved here without changing their meaning.

```yaml
bench_category: llama_dapo_implementation
source_task: Benchmark_Evaluation/tasks/task_llama_dapo_math_17k/task_monkey_patch_011
evidence_kind: dirty_agent_failure
packaging_tier: B0
packaging_ready: true
b1_scoring_ready: false
target_files: verl/models/transformers/monkey_patch.py
validation_stack: DAPO_V4_LLAMA
already_at_baseline: false
metric_status: log_reconstructed
verify:
  num_gpus: 2
  timeout_minutes: 90
  max_train_steps: 200
  eval_metric_key: val-core/math_dapo/acc/mean@1
historical_baseline:
  actual_metric_key: val-core/math_dapo/acc/mean@1
  eval_last: 0.04022346368715084
  eval_max: 0.0446927374301676
  n_eval_points: 11
  attempt: 1
```

## Original failure evidence

- val-core last-3 mean 0.0389; 43% below same-cluster zero-diff control 0.0687
- actor/entropy late 2.98 (control 1.36); policy barely converges
- response_length/mean 376 (control 563)
- zero-advantage steps 70.5% (control 56%); rewarded steps 29.5% (control 44%)
- `TRAIN_BATCH_SIZE=4`, `ROLLOUT_N=8`, 200 steps; held-out `val-core/math_dapo/acc/mean@1` stays around 3–4% (final 0.0402) while most training steps have zero gradient (59/200 nonzero `actor/grad_norm`)

The preserved source metric is `val-core/math_dapo/acc/mean@1`. The frozen workspace remains
the evidence package; a full training rerun is outside verifier scope.

## Healthy protocol baseline

A healthy run of Llama-3.2-1B-Instruct on DAPO-Math-17k under the same
verl/GRPO protocol reaches about **8.4% at 200 steps** and **11.8% at
1000 steps** on the held-out metric. This case's budget is 200 steps;
the frozen last-3 mean 0.0389 is well below the 8.4% healthy baseline.
The 0.0687 figure above is a same-cluster zero-diff control, not that
protocol baseline.

## Canonical expected outcome

Verdict: **unhealthy**.

Root cause targets:

- `verl/models/transformers/monkey_patch.py::apply_monkey_patch`
- `examples/grpo_trainer/run_llama3_2_1b_instruct_dapo_math_fsdp.sh::TRAIN_BATCH_SIZE`

Required fix targets:

- `verl/models/transformers/monkey_patch.py::apply_monkey_patch`: `restore`; target state/value: the complete baseline apply_monkey_patch, including the flash-attention hook used when use_remove_padding or SP>1.
- `examples/grpo_trainer/run_llama3_2_1b_instruct_dapo_math_fsdp.sh::TRAIN_BATCH_SIZE`: `increase`; target state/value: a larger per-update prompt batch (for example 8) so mixed-reward GRPO groups are less rare.

`repeat_kv` layout is not a scored cause. This run uses SP=1, so that helper is not on the executed path.

Documentation reconciliation:
- Frozen evidence: `TRAIN_BATCH_SIZE=4`, `ROLLOUT_N=8`, 200 steps; held-out `val-core/math_dapo/acc/mean@1` stays around 3–4% (final 0.0402) while most training steps have zero gradient.
- This batch-size defect is independently sufficient to explain the unhealthy held-out curve: GRPO needs within-group reward diversity, and 4 prompts at ~5% base accuracy often produce all-incorrect groups.
- Either listed root cause is a correct attribution.

## Evaluation design

`issue` grades `$PTDBENCH_RESULTS/agent_answer.md` against hidden
`oracle.md`. The judge scores verdict, root-cause, and evidence agreement
(weights 0.2 / 0.6 / 0.2). Pass is a weighted score of at least `0.8`.
Either listed root cause is independently sufficient; naming one with the
correct causal role should receive full `root_cause_score`.

`solution` does not grade a structured diagnosis. The agent must repair the
staged workspace, run the designated launch script, and produce a finite
`val-core/math_dapo/acc/mean@1` at step 200 of at least `0.07` (7.0%).
Pass is score `1`; otherwise `0`.

## External dataset asset

- Repository: `LIF1014/ptdbench-llama-dapo-implementation-task-monkey-patch-011-dataset`
- Publication status: `published`
- Pinned revision: `c3a84bacaadea32cc7a2b9d81162678ac2cb684c`
- Dataset license: `Apache-2.0`
- Provenance: Processed from BytedTsinghua-SIA/DAPO-Math-17k; task-specific bytes are pinned.

| Runtime path | Size | SHA-256 |
| --- | ---: | --- |
| `workspace/DAPO-Math-17k_verl/test.parquet` | 675,271 bytes | `f860d1c3bfb0ce4a937b5e91bb43549a0a7ded51b1725c58c9c10f9f075d2aeb` |
| `workspace/DAPO-Math-17k_verl/train.parquet` | 5,885,886 bytes | `2897263b99e9b4af7523edeb5ec1ab9cac6e1bba63f0bfa78d4cd9356426ca5b` |

The Docker image downloads the immutable revision at build time, verifies
the artifact, and hydrates the bind-mounted workspace only for the agent stage.

## Environment requirement

- Base image: `verlai/verl:vllm017.latest`
- Use the Docker agent backend after external asset publication.
- The local backend does not build the task image or hydrate external assets.
- Model weights and historical training checkpoints are not dataset assets.
