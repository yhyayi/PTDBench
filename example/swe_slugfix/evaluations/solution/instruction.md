Codex, fix the bug in `workspace/toy_swe/slugify.py` so the hidden verifier tests pass.

Keep the public module path and function name unchanged:

- `toy_swe.slugify.slugify(text: str) -> str`

The intended behavior is to normalize a short phrase into a stable slug:

- lowercase the text
- replace any run of non-alphanumeric characters with a single `-`
- strip leading and trailing `-`

Only edit files under `workspace/`.
