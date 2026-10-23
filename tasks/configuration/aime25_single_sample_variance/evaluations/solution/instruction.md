# Repair the evaluation protocol

Review the staged AIME25 evaluation and determine whether the single-run result
is reliable. If the evaluation protocol is defective, repair it in the staged
workspace at `/workspace`.

The repaired default protocol must:

- keep the 30-question AIME25 benchmark;
- keep stochastic generation at temperature `1.0` and top-p `1.0`;
- use `32` samples per question;
- aggregate repeated samples with the `mean` reducer.

Concretely, update the workspace evaluation configuration
(`config/eval_config.yaml`) that `run_eval.py` consumes so that
`run_eval.py --protocol-only` reports `samples_per_question: 32` and
`expected_generations: 960`.

Do not modify the preserved original evaluation evidence under
`workspace/ptdbench_case/original_eval/`. Do not modify any evaluation, oracle,
or verifier file. Do not use the network.

## Your job

This task is a protocol repair, not a training run — there is nothing to
launch. **Do not start any evaluation or training run**; cheap, non-training
checks are fine.

When you are done, submit a report at `/workspace/report.md`. It must
summarize:

1. **Root cause** — why the single-run protocol was unreliable.
2. **Changes** — every file you changed and what you changed in it.
3. **Expected effect** — how the repaired protocol addresses the
   unreliability.

A missing or empty `/workspace/report.md` scores 0.

## How you will be graded

After you submit the report, the system runs a task-specific verifier that
deterministically checks the repaired protocol and that the preserved evidence
is unchanged. There is no metric-reconstruction grader and no rerun of an
experiment — the verifier itself is the whole evaluation. See the maintainer
`readme.md`, "Evaluation design".
