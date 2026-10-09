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

> **Note (this task's solution grading differs from the other `ppo_critic`
> tasks).** This is an evaluation-protocol repair, not a training run, so there
> is no metric-reconstruction grader. The verifier deterministically checks the
> repaired protocol and that the preserved evidence is unchanged. See the
> maintainer `readme.md`, "Evaluation design".
