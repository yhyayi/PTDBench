# Repair the rollout stopping configuration

Inspect the historical experiment in `/workspace` and repair its generation
configuration so that completed assistant turns are recognized as an end of
generation. Preserve the existing end-of-text stopping behavior.

This is a configuration-repair variant, not a GPU training challenge. Edit only
`model/generation_config.json`, changing only its `eos_token_id` field. Do not
change model weights, tokenizer metadata, the template, training code, reward
logic, decoding settings or frozen evidence. Do not access the network or launch
training. No performance target or new training log is required.

When you are done, submit a report at `/workspace/report.md`. It must
summarize:

1. **Root cause** — the defect you found, and the evidence in the workspace
   (logs, metrics, code) that connects it to the observed behavior.
2. **Changes** — every file you changed and what you changed in it.
3. **Expected effect** — which metric the fix should move and in which
   direction.

A missing or empty `/workspace/report.md` scores 0.

The submitted artifact is the updated file in the workspace. The verifier reads
that file directly; an explanation or a fabricated log cannot substitute for it.
Its checks validate the EOS set and deterministic stop-boundary semantics, not
actual model inference or an improvement in held-out performance.
