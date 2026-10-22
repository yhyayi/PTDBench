# Provenance of the dense Qwen3-4B model configs

`qwen3-4B.sh`, `qwen3-4B-Thinking-2507.sh` and `qwen3-4B-Instruct-2507.sh` are
**not** from upstream slime and **not** from the OpenRSI release. They are vendored
byte-for-byte from a companion internal repository:

- Repo: `iseesaw/SuperML-Agent` (private)
- Branch: `slime-mle-202602`
- Commit: `bfa16817bea00fa408518aa6bd3535f0d7f9c5fd`
- Path: `slime/scripts/models/`

Upstream slime's released `scripts/models/` (as bundled with OpenRSI) ships only
mixtures — `qwen3-30B-A3B-Thinking-2507.sh` and `qwen3.5-35B-A3B.sh` — so a dense
4B policy config had to come from elsewhere.

`qwen3-4B-Thinking-2507.sh` and `qwen3-4B-Instruct-2507.sh` are one-liners that
re-source `qwen3-4B.sh` with a different `MODEL_ARGS_ROTARY_BASE`; the 2507
variants use `5000000`, base uses the `1000000` default.

If these are later moved upstream, replace this note with the upstream path.
