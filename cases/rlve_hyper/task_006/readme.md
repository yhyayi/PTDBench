# task_006

PTDBench B0 package generated from a Benchmark_Evaluation RLVE historical run.

- Source run: `run_23_rollout_maxlen_512_nsamples_8`
- Failure signature: `rollout_truncation`
- Buggy keys: `rollout_max_response_len`
- Packaging tier: B0 (structure and frozen evidence validated)
- B1: requires local 4-GPU Phase2 execution and metric gate calibration

The Agent sees `workspace/` and `instruction.md`; `oracle/` and `verifier/` are
verifier-only.
