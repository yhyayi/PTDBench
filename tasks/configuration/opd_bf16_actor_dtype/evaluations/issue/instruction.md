# Issue diagnosis

The workspace contains a sampled-token on-policy distillation (OPD) run built on verl: the student
DeepSeek-R1-Distill-Qwen-1.5B is distilled from the teacher JustRL-DeepSeek-1.5B on DAPO-Math-17k.
`on_policy_distillation.sh` is the exact launcher of the run, `verl/` is the training framework,
`logs/` holds the run's console log and its per-step metrics (`logs/metrics.csv`), and `eval/` holds an
offline evaluation of the run's checkpoints. The student and teacher checkpoints are in `model/` and the
data in `datasets/`.

The run trains for hundreds of steps without crashing, and every logged loss, reward and gradient norm
stays finite. **However, the student barely learns.** In this objective the per-token advantage is
`log p_teacher(y_t) - log p_student(y_t)` on the student's sampled tokens, so `-critic/advantages/mean`
is the per-token reverse-KL estimate. A healthy run of this recipe reduces it by roughly an order of
magnitude within about 150 steps, with `actor/grad_norm` falling well below its initial value and the
training accuracy rising. Here all three hardly move, and the offline benchmark scores of the checkpoints
stay flat.

Find the concrete root cause of this behavior.

You may read anything in the workspace and run read-only probes, tests, or short experiments inside it.
Do not modify the workspace for this evaluation, do not access the network, and do not inspect files
outside the workspace and the normal runtime environment.

Write your final report to:

`$PTDBENCH_RESULTS/agent_answer.md`

The report should identify:

* the concrete root cause, down to the setting and the code path that turn it into the observed behavior;
* the conditions that trigger it;
* why the run still looks numerically healthy;
* the evidence in the logs that supports the diagnosis, and which plausible suspects are not responsible;
* an appropriate repair direction.

A report that only says "the learning rate is too small", "the model does not learn" or "bf16 is
imprecise" is insufficient. The diagnosis must explain the mechanism by which the run fails to learn.
