# Configuration repair + replay-eval verifier

Run `bash test.sh`, using `PTDBENCH_WORKSPACE` and `PTDBENCH_RESULTS`
as supplied by the harness. Local testing also accepts `--workspace` and
`--results` arguments. No API credentials are needed.

The verifier has two gates, in order:

1. **Configuration gate** (`grade_submission.py:grade_config`) — checks the
   actual workspace `model/generation_config.json`, rejects symlinks,
   duplicate JSON keys and invalid token identifiers, preserves non-EOS
   fields, and applies the EOS set to synthetic token streams.
2. **Evaluation gate** (`grade_submission.py:grade_eval`) — reads
   `experiment_logs/eval_metrics.json` from the harness's system rerun
   (run_spec.json: `scripts/replay_stopping.py` deterministically replaying
   the retained rollout records, each completion cut at the first stopping
   token read from the repaired `model/generation_config.json` and scored
   with the training's own answer extraction), and requires its accuracy to
   clear `grader_spec.json`'s `required_eval_accuracy`. Calibrated on the
   retained records, this gate is a floor, not the discriminator: the
   repaired config replays at 0.667 with every completion stopping at the
   end of its turn, the unrepaired one at 0.667 with a third of them
   running past the answer to the token cap (see `grader_spec.json`'s
   `reference_calibration`), so discrimination lives in the structural
   gate; this gate catches repairs that pass structurally but degrade
   decoding (a truncating or garbage stop set cuts the scored answers away
   and lands far below the floor).

Both gates must pass (score 1). Results cannot be supplied through
agent-authored logs: the rerun clears `experiment_logs` before decoding, and
the grader only reads the harness-side results root, never the workspace.
It writes `result.json` and `metrics.json` under the results directory and
exits unsuccessfully on a failed check.
