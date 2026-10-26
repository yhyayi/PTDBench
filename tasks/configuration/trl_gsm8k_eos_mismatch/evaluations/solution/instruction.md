# Repair the rollout stopping configuration

Inspect the historical experiment in `/workspace` and repair its generation
configuration so that completed assistant turns are recognized as an end of
generation. Preserve the existing end-of-text stopping behavior.

This is a configuration-repair task, not a training challenge. Edit only
`model/generation_config.json`, changing only its `eos_token_id` field. Do
not change model weights, tokenizer metadata, the template, training code,
reward logic, decoding settings or frozen evidence. Do not access the network
or launch training. No new training run is required — or wanted: training
would take hours and teach nothing here, since the defect is in how finished
turns are recognized, not in what the model learned.

**Do not run the evaluation yourself — do not run
`scripts/replay_stopping.py`, in the foreground or background, and do not use
`nohup`, `&`, `tmux`, `screen`, or any detach mechanism.** The system runs
exactly one evaluation itself after you finish; an evaluation you
start is wasted and scores nothing.

When you are done, submit a report at `/workspace/report.md`. It must
summarize:

1. **Root cause** — the defect you found, and the evidence in the workspace
   (logs, metrics, code) that connects it to the observed behavior.
2. **Changes** — every file you changed and what you changed in it.
3. **Expected effect** — which metric the fix should move and in which
   direction.

A missing or empty `/workspace/report.md` scores 0.

## How you will be checked

After you submit the report, the system runs one evaluation in your repaired
workspace: a deterministic replay of the retained rollout records
(`outputs/rollout_samples.jsonl`), each completion cut at the first stopping
token taken from your `model/generation_config.json` and scored with the
experiment's own answer extraction (the recorded completions, the chat
template and the extraction rule exactly as in the recorded experiment).
The verifier then checks, in order:

1. **The repair itself** — the EOS set in `model/generation_config.json`,
   with the other fields unchanged and deterministic stop-boundary semantics.
2. **What the fix was for** — the replay evaluation's accuracy must clear the
   task's threshold. A configuration that passes the structural checks but
   degrades inference is not a fix.

The submitted artifact is the updated file in the workspace. The verifier
reads that file directly; an explanation or a fabricated log cannot
substitute for it.
