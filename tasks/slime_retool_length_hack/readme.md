# Slime ReTool GRPO run assessment

## Case overview

- Evaluation key: `slime_retool_length_hack`
- Case family: `slime_implementation`
- Schema version: `2`
- Evaluation levels: `issue`, `solution`
- Model: `Qwen3-4B` (Qwen3-4B-Instruct-2507 SFT → RL)
- Dataset: `DAPO-Math-17k` (RL prompts) + `ReTool-SFT` (SFT warm-up)
- Training framework: `slime` (Megatron train + SGLang rollout)
- Recipe: `examples/retool/` — multi-turn code-interpreter (tool-use) GRPO
- Primary source evaluation metric: `eval/aime`

This document is for maintainers and reviewers. It is not exposed to the
agent during evaluation.

## Canonical verdict

**unhealthy.**

This is a defect task: the staged workspace reproduces the canonical ReTool
recipe (SFT → RL with a Python code-interpreter tool) on top of an unmodified
upstream `slime` snapshot, but a length-reward defect has been injected into
`examples/retool/generate_with_retool.py::reward_func`. The reward adds an
unconditional `5 * length_ratio` bonus on top of the correctness score, so the
policy reward-hacks by generating longer rollouts. Frozen historical evidence
through train step 100 shows polluted `eval/aime` rising while true held-out
accuracy collapses.

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
verify:
  num_gpus: 8
  max_train_steps: 100
  eval_interval: 10
  last_eval_step: 99
  eval_metric_key: eval/aime
historical_run:
  metric_status: wandb_extracted
  wandb_run_id: jwszcocp
  wandb_group: qwen3-4B-retool-multi-turn-reward-hack
  eval_first_raw: 0.8217626190185547
  eval_last_raw: 3.8410780588785807
  eval_last_decontaminated: -0.831261698404948
  held_out_accuracy_last: 0.084369150797526
  length_bonus_coefficient: 5.0
  n_eval_points: 11
  attempt: 1
```

The workspace preserves the SFT and RL launch scripts, the custom multi-turn
tool-calling rollout, the sandboxed Python code interpreter, the injected
length-reward defect, a compact frozen metrics package under
`workspace/outputs/`, and a slimmed offline W&B run (console `output.log`
through train step 100, config, summary; no `.wandb` binary history).

## Original failure evidence

- eleven AIME-2024 eval points from step 0 to step 99
- logged `eval/aime` rises from about +0.82 to +3.841 (outside `[-1, 1]`)
- eval response length grows from about 4.1k to 15.3k; eval truncation rises
  from about 0.8% to 82.5%
- train response length at step 100 is about 15159 tokens with a 78.5%
  truncate ratio
- decontaminated held-out accuracy falls from about 29.2% at step 0 to 8.4%
  at step 99

The preserved source metric is `eval/aime`. Because that metric is the mean of
`reward_func`, it includes `5 * length_ratio`. The frozen workspace remains the
evidence package for diagnosis. Solution grading requires the agent to repair
the workspace and rerun training; the verifier reads only the new logs under
`$PTDBENCH_RESULTS/experiment_logs`.

## Why the recipe is unhealthy

Apart from the reward, the ReTool recipe is reproduced correctly, so the defect
must be isolated to the reward rather than the rollout or the algorithm config.
The injected fault is in `generate_with_retool.py::reward_func`:

- **Reward defect** (`generate_with_retool.py::reward_func`): on top of the
  math_dapo exact-match score (`strict_box_verify=True` on `Answer: \boxed{...}`)
  and the bounded tool-call shaping term, the function adds
  `5 * min(sample.response_length / args.rollout_max_response_len, 1.0)` to
  `result["score"]` for **every** sample, unconditionally. Because this bonus
  depends only on generation length and not on correctness, a wrong-but-long
  rollout can outscore a correct-but-short one. Under GRPO the resulting
  advantages reward longer generations, so response length inflates and the
  policy reward-hacks the length term instead of learning to answer correctly.

The surrounding recipe is otherwise correct — these are not the defect:

- **Multi-turn tool rollout** (`examples/retool/generate_with_retool.py`):
  interpreter observations (`<interpreter>…</interpreter>`) injected between
  turns are appended with `trainable=False` (`loss_mask=0`).
- **Token/log-prob alignment**: missing `output_token_logprobs` aborts the
  sample; per-turn token/log-prob parity is asserted; stale rollout state is
  reset on retry.
- **Context-budget safety**: per-turn `max_new_tokens` is clamped and
  overflowing tool output is trimmed.
- **Algorithm/optimizer** (`examples/retool/retool_qwen3_4b_rl.sh`): standard
  GRPO with Adam `--lr 1e-6` constant schedule.
- **Wiring**: only the documented `--custom-generate-function-path` and
  `--custom-rm-path` hooks; the core framework is unmodified.

The length-reward term is the defect. Unusual-looking but correct choices
(`kl-loss-coef 0.00`, `--rotary-base 5000000`, `max_turns=16`,
`tool_concurrency=32`, W&B debug logging, runtime asset paths under `/root/…`)
are expected properties of the recipe and must not be reported as the cause.

## Canonical expected outcome

Verdict: **unhealthy**.

- Root cause target: the unconditional length-reward term in
  `generate_with_retool.py::reward_func` (`length_reward`).
- Required fix target: remove the length-proportional term so the reward depends
  only on correctness plus the bounded tool-call shaping.

An agent that flips the verdict to healthy, treats the rising logged
`eval/aime` as held-out improvement, or attributes the failure to an
unusual-but-benign setting instead of the length-reward term, is wrong.

## Evaluation design

`issue` grades `$PTDBENCH_RESULTS/agent_answer.md` against hidden
`oracle.md`. The judge scores verdict, root-cause, and evidence agreement
(weights 0.2 / 0.6 / 0.2). Pass is a weighted score of at least `0.8`.

`solution` does not grade a structured diagnosis. The agent must edit the
workspace, run `examples/retool/retool_qwen3_4b_rl.sh`, and produce a finite
`eval/aime` at step 99 of at least `0.18` (18%). Pass is score `1`; otherwise
`0`. New runs log `eval/aime` as exact-match accuracy (`--eval-reward-key acc`).
The historical decontaminated accuracy is about 8.4%; a healthy protocol
reaches about 25% at the same horizon.

## Environment requirement

The task runs on the unified PTDBench image, in the `slime-v0.3.1` conda
environment (`environment: conda: slime-v0.3.1`) — the same environment as
`slime_retool`, which is the same recipe. There is no per-task Dockerfile any
more; `environment/prepare.sh` does that work instead.

`prepare.sh` fetches the same four assets as `slime_retool` — DAPO-Math-17k
jsonl, AIME-2024 jsonl, the SFT'd HF checkpoint, and the torch_dist conversion
of it. The two files are deliberately kept identical: this task is the
bug-injected twin of `slime_retool`, so the only difference between them must
be the injected defect. Edit one, edit both.

The workspace code and the ReTool recipe are distributed under Apache-2.0.
