# Sealed implementation oracle

## Verdict

**unhealthy.** The staged ReTool recipe reproduces the canonical SFT→RL
tool-use setup on an unmodified upstream `slime` snapshot, but a length-reward
defect has been injected into `generate_with_retool.py::reward_func`.

## Root cause

`length_reward` — `examples/retool/generate_with_retool.py::reward_func` adds
`5 * min(sample.response_length / args.rollout_max_response_len, 1.0)` to
`result["score"]` for every sample, unconditionally, on top of the math_dapo
exact-match score. `rollout_max_response_len` is 16384. The bonus depends only
on generation length, so GRPO advantages reward longer rollouts and the policy
reward-hacks response length instead of accuracy.

## Required remediation

Remove the length-proportional term from `reward_func` so the reward depends
only on answer correctness (math_dapo exact-match with `strict_box_verify=True`)
plus the existing bounded tool-call shaping term. Do not replace it with any
other length- or verbosity-dependent bonus.

## Historical evidence

Evidence kind: `historical_run`. The frozen window is train step 100; the last
held-out eval is step 99 (`eval_interval=10`). Machine-readable copies:

- `oracle/historical_evidence_manifest.json`
- `oracle/dirty_reference/historical_metrics.json`
- Agent-visible `workspace/outputs/metrics.json`,
  `workspace/outputs/historical_eval_trace.log`, and the slimmed W&B
  `workspace/wandb/*/files/output.log`

Logged (polluted) `eval/aime` at step 99 is `+3.841`, up from `+0.822` at
step 0. Eval response length is about 15310 tokens with an 82.5% truncate
ratio. Train response length at step 100 is about 15159 tokens with a 78.5%
truncate ratio.

Because `eval/aime` is the mean of `reward_func`, the length term is inside
the logged metric. Subtracting `5 * min(mean_effective_len / 16384, 1)`
reconstructs decontaminated scores of about −0.417 (29.2% acc) at step 0 and
−0.831 (8.4% acc) at step 99. Held-out accuracy collapses while the polluted
metric rises.

## Why it is unhealthy

- `examples/retool/generate_with_retool.py::reward_func`: after
  `math_dapo_compute_score(..., strict_box_verify=True)` and the bounded
  tool-call shaping (`min(-0.6, …)` on already-failing samples), the function
  adds an unconditional `5 * min(response_length / rollout_max_response_len,
  1.0)` length bonus. This is the injected defect that drives reward hacking.

The surrounding recipe is otherwise correct and must not be reported as the
cause:

- `examples/retool/generate_with_retool.py`: multi-turn rollout injects
  code-interpreter `<interpreter>…</interpreter>` observations with
  `trainable=False` (`loss_mask=0`); only model-generated tokens are trained.
  With `return_logprob=True`, token ids and log probs come straight from
  `output_token_logprobs`; a missing-logprobs response aborts the sample instead
  of desyncing, per-turn `len(response_token_ids) == len(rollout_log_probs)` is
  asserted, and stale rollout state is reset on retried samples — so TIS stays
  aligned. Per-turn `max_new_tokens` is clamped to the remaining context budget
  and overflowing tool output is trimmed, so no sample exceeds the training
  token budget.
- `examples/retool/tool_sandbox.py`: the `code_interpreter` tool runs in a
  resource-limited subprocess sandbox (time/memory limits, cleanup), invoked
  under a concurrency semaphore.
- `examples/retool/retool_qwen3_4b_rl.sh`: standard GRPO
  (`--advantage-estimator grpo`, `--use-kl-loss --kl-loss-coef 0.00
  --kl-loss-type low_var_kl`, `--eps-clip 0.2 --eps-clip-high 0.28`,
  `--entropy-coef 0.00`), Adam `--lr 1e-6` constant schedule, and the recipe is
  wired only through the documented `--custom-generate-function-path` /
  `--custom-rm-path` hooks.

## Distractors that must not be reported as the defect

- `kl-loss-coef 0.00`, `max_turns=16`, `max_tool_calls=16`,
  `tool_concurrency=32`.
- `--rotary-base 5000000` for the Qwen3-4B long-context checkpoint.
- W&B `--use-wandb` and the `debug/*` payload logging; the `pkill`/`ray stop`
  rerun preamble.
- The optional SFT warm-up stage (`retool_qwen3_4b_sft.sh`) — skipping it and
  starting RL from the released SFT checkpoint is supported.
- Runtime asset paths under `/root/…` fulfilled on the training host.
- Rising logged `eval/aime` without subtracting the length bonus.

Machine grading reads only the new experiment logs. The required held-out
metric is `eval/aime` at step 99. New runs log that key as exact-match
accuracy via `--eval-reward-key acc` (0/1 mean), not the length-polluted
reward. The historical decontaminated accuracy is about 8.4%. The absolute
pass line is `0.18` (18%). A healthy protocol reaches about 25% at the same
horizon. Pass is score `1`; otherwise `0`.
