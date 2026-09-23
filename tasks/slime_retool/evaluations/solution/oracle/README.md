# Sealed implementation oracle

## Verdict

**healthy.** The staged ReTool recipe is a faithful reproduction of the
canonical SFT→RL tool-use setup on an unmodified upstream `slime` snapshot. No
defect has been injected.

## Root cause

None. The expected `root_cause` array is empty.

## Required remediation

None. The expected `fix` array is empty. A healthy run requires no change.

## Historical evidence

Evidence kind: `historical_run`. The frozen window is train step 100; the last
held-out eval is step 99 (`eval_interval=10`). Machine-readable copies:

- `oracle/historical_evidence_manifest.json`
- `oracle/dirty_reference/historical_metrics.json`
- Agent-visible `workspace/outputs/metrics.json`,
  `workspace/outputs/historical_eval_trace.log`, and the slimmed W&B
  `workspace/wandb/*/files/output.log`

Primary metric `eval/aime` at step 99 is `-0.4965625` (about 25.2% accuracy),
up from `-0.59125` (20.4%) at step 0. Train response length at step 100 is
about 6394 tokens with a 5.1% truncate ratio.

## Why it is healthy

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
- `examples/retool/generate_with_retool.py::reward_func`: math_dapo exact-match
  with `strict_box_verify=True` on `Answer: \boxed{...}`, plus a bounded
  tool-call shaping term that only nudges already-failing samples. Length does
  not add reward, so logged `eval/aime` is a true held-out score.
- `examples/retool/retool_qwen3_4b_rl.sh`: standard GRPO
  (`--advantage-estimator grpo`, `--use-kl-loss --kl-loss-coef 0.00
  --kl-loss-type low_var_kl`, `--eps-clip 0.2 --eps-clip-high 0.28`,
  `--entropy-coef 0.00`), Adam `--lr 1e-6` constant schedule, and the recipe is
  wired only through the documented `--custom-generate-function-path` /
  `--custom-rm-path` hooks.

## Distractors that must not be reported as defects

- `kl-loss-coef 0.00`, `max_turns=16`, `max_tool_calls=16`,
  `tool_concurrency=32`.
- `--rotary-base 5000000` for the Qwen3-4B long-context checkpoint.
- W&B `--use-wandb` and the `debug/*` payload logging; the `pkill`/`ray stop`
  rerun preamble.
- The optional SFT warm-up stage (`retool_qwen3_4b_sft.sh`) — skipping it and
  starting RL from the released SFT checkpoint is supported.
- Runtime asset paths under `/root/…` fulfilled on the training host.
- Moderate length growth and a 22% eval truncate ratio at step 99. These are
  not a collapse and are not caused by a length bonus.

Machine grading at the solution level is deterministic and metric-only: the
verifier reconstructs `eval/aime` from the logs the agent's own run wrote under
`$PTDBENCH_RESULTS/experiment_logs` and compares it to a fixed threshold. It
does not read `gold.yaml`, which is reviewer-facing documentation. See
`../verifier/README.md` for the threshold and its calibration.
