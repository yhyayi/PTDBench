# Qwen DAPO hparam diagnosis: task_hparam_seqlen_micro_014

## Case overview

- Evaluation key: `verl_hyper/qwen_dapo_hparam/task_hparam_seqlen_micro_014`
- Case family: `qwen_dapo_hparam`
- Schema version: `2`
- Evaluation levels: `issue`, `solution`
- Model: `Qwen2.5-0.5B-Instruct`
- Dataset: `DAPO-Math-17k`
- Training framework: `verl`
- Primary source evaluation metric: `val-core/math_dapo/acc/mean@1`

This document is for maintainers and reviewers. It is not exposed to the
agent during evaluation. Metric names preserve the source experiment;
tasks using the same held-out protocol may intentionally share a name.

## Canonical failure reason

MAX_PROMPT_LENGTH and MAX_RESPONSE_LENGTH set to 512 instead of 1024 filter prompts, clip 76.6% of responses, and halve val-core versus control

## Workspace and hidden evaluations

| Path | Visibility and purpose |
| --- | --- |
| `workspace/` | Agent-visible historical experiment snapshot; evidence rather than canonical ground truth. |
| `evaluations/issue/` | Neutral diagnosis instruction, issue oracle, and task-local issue verifier. |
| `evaluations/solution/` | Neutral diagnosis/remediation instruction, hidden reference material, and combined verifier. |

The agent cannot read oracle or verifier paths. The issue evaluation uses an
API-backed semantic judge against `oracle.md`; solution grading remains defined
by its verifier gold and implementation.

## Source and preserved context

The following source and historical packaging fields were moved from
`metadata.yaml` so the machine metadata remains aligned with the core
schema. They are preserved here without changing their meaning.

```yaml
bench_category: qwen_dapo_hparam
source_task: Benchmark_Evaluation/tasks/task_qwen_dapo_hparam/task_hparam_seqlen_micro_014
evidence_kind: dirty_agent_failure
packaging_tier: B0
packaging_ready: true
b1_scoring_ready: false
hollow_hparams: MAX_PROMPT_LENGTH,MAX_RESPONSE_LENGTH,PPO_MICRO_BATCH_SIZE_PER_GPU,LOG_PROB_MICRO_BATCH_SIZE_PER_GPU,TRAIN_BATCH_SIZE
verify:
  num_gpus: 2
  timeout_minutes: 90
  max_train_steps: 200
  eval_interval: 20
  eval_metric_key: val-core/math_dapo/acc/mean@1
historical_baseline:
  eval_first: 0.017347509792949075
  eval_last: 0.01510912143256855
  n_eval_points: 11
  attempt: 8
  first_rollout_ok: true
```

## Original failure evidence

- response_length/clip_ratio ≈ 0.766
- prompt_length/max drops from control 745 to 469 while prompt_length/clip_ratio stays 0 (filtering, not truncation)
- val-core 0.017 → 0.015; about 1/2.3 of the control

The preserved source metric is `val-core/math_dapo/acc/mean@1`. The frozen workspace remains
the evidence package; a full training rerun is outside verifier scope.

## Healthy protocol baseline

A healthy run of Qwen2.5-0.5B-Instruct on DAPO-Math-17k under the same
verl/GRPO protocol reaches about **6.0% at 200 steps** and **7.7% at
1000 steps** on `val-core/math_dapo/acc/mean@1`. This case's budget is
200 steps; the frozen `eval_last` 0.0151 is well below the 6.0% healthy
baseline. The "1/2.3 of the control" figure above is a same-cluster
comparison, not that protocol baseline.

## Canonical expected outcome

Verdict: **unhealthy**.

Root cause targets:

- `MAX_RESPONSE_LENGTH`
- `MAX_PROMPT_LENGTH`

Required fix targets:

- `MAX_RESPONSE_LENGTH`: `increase`; target state/value: 1024.
- `MAX_PROMPT_LENGTH`: `increase`; target state/value: 1024.

Documentation reconciliation:
- The three micro-batch fields preserved in hollow_hparams are distractors; canonical evidence identifies only the two sequence-length limits.

## Evaluation design

The issue level requires a concise diagnosis in
`$PTDBENCH_RESULTS/agent_answer.md`; its API-backed judge scores verdict,
root-cause attribution, and evidence against the hidden `oracle.md`.

The solution level continues to require `$PTDBENCH_RESULTS/submission.json`
with `verdict`, `root_cause`, `evidence`, and `fix`. It applies the issue gate
and grades the proposed remediation.

- Combined score: `issue_score * (0.5 + 0.5 * solution_score)`.
- Statuses: `correct`, `partial`, `wrong`, or `unavailable`.

The prompts are neutral and contain no task-specific answer. Documentation
generation does not independently alter verifier code or scoring; reviewed
gold changes require the explicit `align-verifier-gold` command.

## External dataset asset

- Repository: `LIF1014/ptdbench-qwen-dapo-hparam-task-hparam-seqlen-micro-014-dataset`
- Publication status: `published`
- Pinned revision: `784a6b8c4f7270e05bed48d1a7cd109abcfa7cc2`
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
