# Repair the OPD run

The sampled-token on-policy distillation run in this workspace (`on_policy_distillation.sh`, see `logs/`
and `eval/`) completes without errors, but the student barely learns: the per-token reverse KL
(`-critic/advantages/mean`), `actor/grad_norm` and the training accuracy hardly move, and the offline
benchmark scores of its checkpoints stay flat. Diagnose and repair the workspace so that the run trains
properly.

Constraints:

* Keep the training recipe: the student and teacher checkpoints, the training data, the sampled-token
  OPD objective (`ADV_ESTIMATOR=token_reward_direct`, `LOG_PROB_TOP_K=0`), the optimizer and its
  hyperparameters (AdamW, learning rate `1e-6`, weight decay, gradient clipping) and the batch layout.
  Fixes that only change these settings are not accepted.
* `on_policy_distillation.sh` must remain the training entrypoint, launch on the GPUs listed in
  `CUDA_VISIBLE_DEVICES`, and keep forwarding its extra command-line arguments to
  `verl.trainer.main_ppo`. The verifier launches a shortened run through it.
* Do not edit evaluation files, do not special-case the verifier, and do not access the network.

You may run the training script or smaller experiments to check your repair. A full-length run is not
required.
