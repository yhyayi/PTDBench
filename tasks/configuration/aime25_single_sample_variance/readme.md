# PPO-Critic run assessment: aime25_single_sample_variance

## Case overview

- Evaluation key: `aime25_single_sample_variance`
- Case family: `ppo_critic`
- Schema version: `2`
- Evaluation levels: `issue`, `solution`
- Model: `Qwen3-4B-Base`
- Dataset: `AIME25` (30 questions)
- Harness: `verl` PPO-Critic checkpoint, evaluated via `run_eval.py`
- Entry point: `run_eval.py` (reads `config/eval_config.yaml`)

This document is for maintainers and reviewers. It is not exposed to the agent
during evaluation.

## Canonical verdict

**unhealthy evaluation (false alarm), not a training failure.**

The apparent AIME25 "failure" is an artifact of an under-sampled evaluation
protocol, not a defect in PPO training. `config/eval_config.yaml` sets
`evaluation.samples_per_question: 1`: a single stochastic generation per
question (temperature `1.0`, top-p `1.0`) on a benchmark of only 30 questions.
One 30-question realization is therefore a high-variance estimator, and a single
low score is not evidence that PPO training itself failed.

Controlled repeated evaluation of the *same* checkpoint (preserved under
`workspace/ptdbench_case/original_eval/`) produced per-epoch AIME25 accuracy
ranging from **6.67% to 20%**, with **10%** appearing in **7 of 32** repeated
epochs.

## Workspace and hidden evaluations

| Path | Visibility and purpose |
| --- | --- |
| `workspace/` | Agent-visible snapshot: `run_eval.py`, `config/eval_config.yaml` (the buggy `samples_per_question: 1` protocol), and the source tree. |
| `workspace/ptdbench_case/original_eval/` | Preserved controlled evidence: `n1/` (single-sample run), `n32/` (32-sample run), and `notes/` (epoch breakdown, experiment design). Frozen; must not change. |
| `evaluations/issue/` | Neutral diagnosis instruction, issue oracle, and task-local LLM-judge verifier. |
| `evaluations/solution/` | Protocol-repair instruction, hidden reference material, and a deterministic protocol-check verifier. |

The agent cannot read oracle or verifier paths. `issue` grading compares
`$PTDBENCH_RESULTS/agent_answer.md` to hidden `oracle.md` with an LLM judge.

## Original failure evidence

From `workspace/ptdbench_case/original_eval/`:

- `n1/` — one stochastic sample per question; a single 30-question realization.
- `n32/` — 32 samples per question; `notes/n32_epoch_breakdown.txt` records the
  per-epoch spread (6.67%–20%, 10% in 7/32 epochs).
- `notes/experiment_design.txt` / `notes/final_comparison.txt` — the controlled
  comparison of 1-sample vs 32-sample protocols on the same checkpoint.

The checkpoint itself is a large external resource and is not embedded.

## Why the evaluation is unhealthy

Nothing in the training or the model is shown to be defective. The defect is in
the measurement protocol:

- **Protocol defect** (`config/eval_config.yaml`): `samples_per_question: 1` on
  a 30-question benchmark with stochastic decoding. `run_eval.py` reports this
  protocol via `--protocol-only` (`expected_generations: 30`).

AIME25 having only 30 questions, stochastic decoding (`temperature: 1.0`,
`top_p: 1.0`), and PPO-Critic being compared to another algorithm's reported
score are context, not causes. The cause is drawing a single sample and
treating it as stable.

## Canonical expected outcome

Verdict: **evaluation under-sampling / high-variance false alarm**.

- Root cause target: `evaluation.samples_per_question: 1` in
  `config/eval_config.yaml`.
- Required fix target: raise `samples_per_question` to `32` with the `mean`
  reducer, keeping the 30-question benchmark and stochastic generation
  (`temperature: 1.0`, `top_p: 1.0`). After the fix, `run_eval.py
  --protocol-only` reports `samples_per_question: 32` and
  `expected_generations: 960`.

An agent that attributes the low score to a PPO training failure, a reward
defect, or an optimizer problem is wrong.

## Evaluation design

`issue` grades `$PTDBENCH_RESULTS/agent_answer.md` against hidden `oracle.md`.
The judge scores verdict, root-cause, and evidence agreement (weights
0.2 / 0.6 / 0.2). Pass is a weighted score of at least `0.8`.

> **`solution` is a deterministic protocol check, not a metric grader — this is
> a deliberate exception.** Unlike the other three `ppo_critic` tasks, this task
> fixes an *evaluation configuration*, not a training run, so there is no
> training to rerun and no `val-core` / `critic/rewards` metric to reconstruct.
> `evaluations/solution/verifier/test.sh` instead:
>
> 1. runs `run_eval.py --protocol-only` from the repaired workspace and asserts
>    the active protocol is `task == aime25`, `questions == 30`,
>    `samples_per_question == 32`, `reducer == mean`, `temperature == 1.0`,
>    `top_p == 1.0`, and `expected_generations == 960`; and
> 2. verifies the preserved evidence under
>    `workspace/ptdbench_case/original_eval/` is byte-for-byte unchanged against
>    the trusted manifest `oracle/original_eval.sha256` (`sha256sum -c`).
>
> This check is exact and fully runnable offline (no GPU, no network, no model),
> so unlike the training tasks it needs no uncalibrated threshold. `oracle/`
> carries the repaired `eval_config.yaml` (`samples_per_question: 32`),
> `expected.md`, and `original_eval.sha256`. The verifier reads
> `$PTDBENCH_ORACLE`, `$PTDBENCH_WORKSPACE`, and `$PTDBENCH_RESULTS`.

## Environment

`metadata.environment.image: vllm0.11.0-torch2.8.0-flashattn2.8.3-cu128-py3.12` resolves to the shared image
`docker/images/vllm0.11.0-torch2.8.0-flashattn2.8.3-cu128-py3.12.Dockerfile`, the same image the three ppo_critic
training tasks use. This task rides it only for a Python interpreter: its two
graded steps (`run_eval.py --protocol-only` and the evidence `sha256sum -c`) run
with the standard library alone, never touch the rollout/eval stack, and the
preserved evaluation evidence ships inside
`workspace/ptdbench_case/original_eval/`. `environment/prepare.sh` is therefore
a documented no-op (it fetches nothing and records why): neither graded step
loads a model, and the evidence is embedded. The optional full AIME25 rerun
takes an external PPO-Critic checkpoint via `CHECKPOINT_PATH` — a trained
checkpoint, not a fetchable HF base model — supplied by the operator, which the
grading does not depend on.

The *recorded* AIME25 evaluation was produced on a separate eval stack (CUDA
12.6.3 / torch 2.7.1 / vLLM 0.10.1.1 with inspect-ai + lighteval, pinned in the
bundle under `tasks/PTDBench_candidate_bundle_20260821/.../environment/`); that
stack is intentionally **not** rebuilt here, since no checkpoint is embedded and
the grading is stdlib-only. The full AIME25 rerun (optional) needs a PPO-Critic
checkpoint supplied via `CHECKPOINT_PATH`, a large external resource not embedded
here.

The workspace code and the verl snapshot are distributed under Apache-2.0; the
task packaging is MIT.
