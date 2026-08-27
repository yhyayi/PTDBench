# Slime Search-R1 GRPO run assessment: slime_search_r1_nomask

## Case overview

- Evaluation key: `slime_search_r1_nomask`
- Case family: `slime_implementation`
- Schema version: `2`
- Evaluation levels: `issue`, `solution`
- Model: `Qwen2.5-3B`
- Dataset: `Search-R1 nq_hotpotqa_train` (NQ + HotpotQA merged)
- Training framework: `slime` (Megatron train + SGLang rollout)
- Recipe: `examples/search-r1/` — multi-turn retrieval-augmented GRPO
- Primary source evaluation metric: `eval/nq_test`

This document is for maintainers and reviewers. It is not exposed to the
agent during evaluation. Metric names preserve the source experiment;
tasks using the same held-out protocol may intentionally share a name.

## Canonical failure reason

Observation tokens are trained on (`loss_mask=1` on retrieved
`<information>` text); frozen `eval/nq_test` reaches only 14.5% at the
100-step horizon versus a healthy protocol baseline of 37.5%.

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

```yaml
evidence_kind: historical_run
packaging_ready: true
healthy_control: tasks/slime_search_r1
verify:
  num_gpus: 8
  max_train_steps: 100
  eval_interval: 10
  last_eval_step: 99
  eval_metric_key: eval/nq_test
historical_run:
  metric_status: wandb_extracted
  wandb_run_id: wi4fh8cz
  wandb_group: search-r1_qwen2.5-3B-nomask
  eval_first: 0.043
  eval_last: 0.145
  eval_max: 0.15266666666666667
  n_eval_points: 10
  attempt: 1
healthy_control_at_horizon:
  wandb_eval_step: 99
  eval/nq_test: 0.37533333333333335
```

The workspace preserves the launch script, the custom multi-turn search
rollout with the injected observation-mask defect, the exact-match reward,
a compact frozen metrics package under `workspace/outputs/`, and a slimmed
offline W&B run (console `output.log` through train step 100, config,
summary; no `.wandb` binary history).

## Original failure evidence

- ten reconstructed `eval/nq_test` points through `eval/step` 99 (the last
  eval in `--eval-interval 10` before the 100-step horizon) rise from about
  4.3% to 14.5% (`eval_last` 0.145; peak 15.3% at step 89)
- a healthy Qwen2.5-3B Search-R1 GRPO run (`tasks/slime_search_r1`) reaches
  about **37.5%** at the same horizon (`eval/nq_test` 0.3753 at `eval/step` 99)
- training `rollout/raw_reward` at step 99 is 0.1875 and does not establish
  held-out health
- truncation falls from ~39% at step 0 to ~0.1% by step 99, so the held-out
  gap is not a truncation artifact
- `examples/search-r1/generate_with_search.py` appends retrieved
  `<information>` tokens with `loss_mask=1` while `append_response_tokens`
  still marks them `trainable=False`; the local list is assigned to
  `sample.loss_mask`

The preserved source metric is `eval/nq_test`. The frozen workspace remains
the evidence package for diagnosis. Solution grading requires the agent to
repair the workspace and rerun training; the verifier reads only the new
logs under `$PTDBENCH_RESULTS/experiment_logs`.

## Healthy protocol baseline

A healthy run of Qwen2.5-3B on Search-R1 nq_hotpotqa_train under the same
slime/GRPO recipe, with observation tokens masked out of the loss, reaches
about **37.5% at 100 steps**. This case's frozen `eval_last` 14.5% is well
below that protocol baseline.

## Canonical expected outcome

Verdict: **unhealthy**.

Root cause targets:

- `observation_loss_mask` —
  `examples/search-r1/generate_with_search.py::generate`

Required fix targets:

- `observation_loss_mask`: `set` observation tokens out of the loss
  (`loss_mask += [0] * len(obs_tokens_ids)`).

Documentation reconciliation:

- Frozen W&B reconstruction uses `eval/step` 99 as the 100-step horizon
  because slime logs `eval/step = rollout_id` and evals at 0, 9, …, 99.
- Eval 19 is missing from the captured `output.log` fragments; the ten
  remaining points are enough to reconstruct the curve.
- Naming the observation loss mask with the correct causal role is a
  complete attribution.

## Evaluation design

`issue` grades `$PTDBENCH_RESULTS/agent_answer.md` against hidden
`oracle.md`. The judge scores verdict, root-cause, and evidence agreement
(weights 0.2 / 0.6 / 0.2). Pass is a weighted score of at least `0.8`.

`solution` does not grade a structured diagnosis. The agent must edit the
workspace, run `examples/search-r1/run_qwen2.5_3B.sh`, and produce a finite
`eval/nq_test` of at least `0.28` (28%). Pass is score `1`; otherwise `0`.
The historical final baseline is `0.145`; the healthy protocol is about
37.5% at the same horizon.

## Environment requirement

- Base image: `slimerl/slime:latest`
- Use the Docker agent backend.
- The image does **not** download the Qwen2.5-3B checkpoint, the
  `nq_hotpotqa_train` Parquet data, or the ~130 GB wiki-18 FAISS retrieval
  index. Issue diagnosis uses the frozen historical snapshot. Solution
  reruns still need those host-side assets already present under
  `/workspace` and a local retrieval server on `:8000`.
- Model weights and historical training checkpoints are not dataset assets.

The workspace code and the Search-R1 recipe are distributed under Apache-2.0.
