# Packaging record

Maintainer-only (the agent stage mounts `workspace/`, the instruction and
`environment/`; the verifier stage sees the task root, which is why nothing here
may be read by the agent).

- `status.json` — what was verified on the packaging host and what was not, with
  the next step for each unverified item. Read this before trusting any number
  in `readme.md` that is not quoted from a log.
- `stage_evidence.py` — slims `results/baseline32/` from 37 MB to ~3.2 MB and
  rewrites the strings that named the answer (the recorded run lived in a
  directory called `broken`, beside one called `fixed`). Writes a substitution
  report to `/tmp/ptdbench_stage/evidence_report.json`, which
  `make_source_manifest.py` folds into `environment/source_manifest.json`.
- `make_data_manifest.py` — builds `environment/data_manifest.json` from the
  delivery's own `shared/data/manifest.json` plus the shipped files' hashes, and
  asserts the two agree.
- `make_oracle_metrics.py` — builds
  `evaluations/solution/oracle/historical_evidence_manifest.json` and
  `oracle/dirty_reference/historical_metrics.json` from the delivery's
  `comparison.json` and the per-step `rollouts/metrics_*.json`.
- `make_source_manifest.py` — builds `environment/source_manifest.json`:
  what was copied byte for byte, what was adjusted for packaging and why, what
  was dropped, and every evidence substitution with before/after hashes.

All four take the delivery at `raw_tasks/hsz_search-r1-delivery` (git-ignored) as
input and are idempotent. Run order matters only for the last one:
`stage_evidence.py` must run before `make_source_manifest.py`, which reads its
report.
