# Qwen DAPO response-length cap diagnosis

## Case overview

- Evaluation key: `verl_hyper/qwen_dapo_hparam/task_hparam_overlong_023`
- Case family: `qwen_dapo_hparam`
- Schema version: `2`
- Evaluation levels: `issue`, `solution`
- Model: `Qwen2.5-0.5B-Instruct`
- Dataset: `DAPO-Math-17k`
- Training framework: `verl`
- Primary source evaluation metric: `val-core/math_dapo/acc/mean@1`

This document is for maintainers and reviewers. It is not exposed to the
agent during evaluation.

## Canonical failure reason

An effective response-length cap of 256 clips nearly all rollouts and coincides with sparse rewards and declining held-out accuracy

## Workspace and hidden evaluations

| Path | Visibility and purpose |
| --- | --- |
| `workspace/` | Agent-visible historical experiment snapshot; evidence rather than canonical ground truth. |
| `evaluations/issue/` | Diagnosis instruction, issue oracle, and task-local issue verifier. |
| `evaluations/solution/` | Diagnosis/remediation instruction, hidden reference material, and combined verifier. |

The agent cannot read oracle or verifier paths. Both verifiers contain
byte-identical gold copied from `evaluation/testset`. If legacy
`standard_answer.txt` differs, canonical evaluation gold controls scoring.

## Source and preserved context

- Source task:
  `Benchmark_Evaluation/tasks/task_qwen_dapo_hparam/task_hparam_overlong_023`
- Failure source: an unsuccessful agent-authored training configuration.
- Historical attempt: 9.
- Training budget: 200 steps on 2 GPUs, with held-out evaluation every
  20 steps.
- Primary held-out metric: `val-core/math_dapo/acc/mean@1`.

The workspace preserves the launch script, effective Hydra configuration,
training logs, W&B offline artifacts, generated metrics, reward stack, and
historical evidence needed to diagnose the run.

## Original failure evidence

The first rollout completed, but the held-out score declined from approximately
`0.001676` to `0.001117` across 11 evaluation points. Almost all generated
responses reached the configured maximum length, while only about 5% of steps
received a reward.

The frozen workspace is the reproducible evidence package. A full training
rerun is not part of the verifier and would additionally require the
Qwen2.5-0.5B-Instruct checkpoint and suitable GPUs.

## Healthy protocol baseline

A healthy run of Qwen2.5-0.5B-Instruct on DAPO-Math-17k under the same
verl/GRPO protocol reaches about **6.0% at 200 steps** and **7.7% at
1000 steps** on `val-core/math_dapo/acc/mean@1`. This case's budget is
200 steps; the frozen `eval_last` 0.001117 is well below the 6.0%
healthy baseline.

## Canonical expected outcome

Verdict: **unhealthy**.

Root cause targets:

- `MAX_RESPONSE_LENGTH`

Required fix targets:

- `MAX_RESPONSE_LENGTH`: `increase`

The inactive overlong-buffer fields are distractors: their enable gate is off,
so changing those values does not address the observed clipping.

## Evaluation design

Both levels require `$PTDBENCH_RESULTS/submission.json` with `verdict`,
`root_cause`, `evidence`, and `fix`.

- `issue` grades verdict and root-cause attribution.
- `solution` applies the issue gate and grades the proposed remediation.
- Combined score: `issue_score * (0.5 + 0.5 * solution_score)`.
- Statuses: `correct`, `partial`, `wrong`, or `unavailable`.

The verifier is rule-first and may use its existing LLM-judge fallback for
semantically equivalent wording. It evaluates the structured diagnosis and
remediation against hidden canonical gold; it does not launch training or
require network access.

## External dataset asset

The case-specific `DAPO-Math-17k_verl` snapshot is stored in the public
Hugging Face dataset repository:

- Repository: `LIF1014/ptdbench-task-hparam-overlong-023-dataset`
- Source URL:
  `https://huggingface.co/datasets/LIF1014/ptdbench-task-hparam-overlong-023-dataset`
- Pinned revision: `ab385ea0b4fd9a8c16680917660d442121a6b5eb`
- License: Apache-2.0

Reconstructed files:

| Runtime path | Size | SHA-256 |
| --- | ---: | --- |
| `workspace/DAPO-Math-17k_verl/train.parquet` | 5,885,886 bytes | `2897263b99e9b4af7523edeb5ec1ab9cac6e1bba63f0bfa78d4cd9356426ca5b` |
| `workspace/DAPO-Math-17k_verl/test.parquet` | 675,271 bytes | `f860d1c3bfb0ce4a937b5e91bb43549a0a7ded51b1725c58c9c10f9f075d2aeb` |

`environment/Dockerfile` downloads the revision-pinned Base64 artifacts,
reconstructs both Parquet files, and verifies their hashes and sizes during
the image build. Because PTDBench bind-mounts the staged workspace over
`/workspace`, the image stores verified copies under
`/opt/ptdbench-assets/workspace/` and restores them into the mounted workspace
when the agent container starts. Evaluation therefore requires no runtime
network access.

## Environment requirement

Use the Docker agent backend for this case. The local backend does not build
the task image and therefore does not materialize the external dataset into
the workspace.

The workspace code and the external processed dataset are distributed under
Apache-2.0.
