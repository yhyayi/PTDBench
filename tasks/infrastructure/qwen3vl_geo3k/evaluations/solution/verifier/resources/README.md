# Grader inputs

Run the parent `test.sh`, not an agent-supplied metrics file. Fresh experiment
logs must be below `$PTDBENCH_RESULTS/experiment_logs`.

`grader_spec.json` defines the metric
`per_response_token_logprob_abs_diff_nat`, six cases, 64 response tokens,
1.0 nat repair limit, and 20 required training updates. Correct positions
are mandatory. The severe-candidate 0.25 nat and paired 2x limits belong to
the paired reference check. They are not solution-score requirements or
reward-improvement targets.
