# User input example

Provide everything in this structure in one go when packaging. If any item is missing, stop. Replace numbers, commits, and hashes with the real values from that training run.

When packaging, read the code and launch-script diff between the buggy commit and the baseline commit, write the root cause as one sentence, and include the metric symptoms given here.

## Evaluation score only

```yaml
source_workspace: /home/test/test06/lyh/verl
output_dir: tasks/infrastructure/task_example_001
task_id: task_example_001
title: "Missing flash-attention hook"
type: infrastructure          # the category = the tasks/<type>/ directory (one of the six)
domain: post_train            # free-form descriptive label; may repeat the type
model_name: Llama-3.2-1B-Instruct
dataset_name: DAPO-Math-17k
framework: verl
algo: grpo

git_repo: /home/test/test06/lyh/verl
buggy_commit: aaaa1111
baseline_commit: bbbb2222
ignore_hunks: []

launch_script: examples/grpo_trainer/run_llama3_2_1b_instruct_dapo_math_fsdp.sh
launch_args: []
launch_env:
  WANDB_MODE: offline
model_path: /workspace/verl/Llama-3.2-1B-instruct
train_path: /workspace/DAPO-Math-17k_verl/train.parquet
val_path: /workspace/DAPO-Math-17k_verl/test.parquet
failure_logs:
  - logs/llama3_2_1b_instruct_dapo_math_fsdp_20260720_005502.log
include_wandb: true

image: verl
gpus: "0,1"
needs_compose: false

assets:
  - kind: hf
    repo: LIF1014/ptdbench-Llama-3.2-1B-Instruct
    type: model
    revision: 32f9e721f796d3c639ab64d2f58ceeaa2c3213b9
    allow:
      - config.json
      - generation_config.json
      - tokenizer_config.json
      - tokenizer.json
      - tokenizer.model
      - model.safetensors
    expect:
      model.safetensors: 1ff795ff6a07e6a68085d206fb84417da2f083f68391c2843cd2b8ac6df8538f
    expect_size:
      model.safetensors: 2471645608
    dest: verl/Llama-3.2-1B-instruct
    why_this_path: >-
      Launch script :30-38 searches /models/… → /workspace/verl/Llama-3.2-1B-instruct
      in order. Nothing mounts the first candidate, so it must land exactly on the
      second. The directory name uses a lowercase i.
  - kind: recipe
    recipe: dapo_math_17k
    expect:
      train.parquet: fc4cad54f2d15c622025b5e3764e91d2a79145e75358a703658daa66fe47d321
      test.parquet: 52596fd6ade0d49db3a49ef0c9c89f88aeb323c17fa5ee23fb0c774d7fba3255
    expect_size:
      train.parquet: 2735462
      test.parquet: 331284
    dest: DAPO-Math-17k_verl

metric_key: val-core/math_dapo/acc/mean@1
metric_source: text_log
log_sample: "step:200 - val-core/math_dapo/acc/mean@1:0.0402"
final_step: 200
historical_final_baseline: "0.04022346368715084"
required_final_score: "0.07"

historical_eval_points:
  - { step: 0, eval_score: 0.030726256983240222 }
  - { step: 100, eval_score: 0.03519553072625698 }
  - { step: 200, eval_score: 0.04022346368715084 }
```

A `failure_reason` written from this diff would look like: `apply_monkey_patch no longer installs the flash-attention hook; final val-core is about 0.0402`. The exact wording follows the filtered diff.

Each entry in `assets` becomes one `ptdbench_fetch_hf` / `ptdbench_dataset` / `ptdbench_fetch_url` call in `environment/prepare.sh`, with `$PTDBENCH_WORKSPACE/` prefixed to `dest`. `why_this_path` becomes the comment in the script, because a wrong path only blows up midway through training.
## No scoring method given

The example above, up to `required_final_score`, is the default scoring: the score depends entirely on the eval-set key. Per `score_mode`, take that key's highest finite value (default `best`) or the value at `final_step` (`final`). If it reaches `required_final_score`, score is 1; otherwise 0.

Give these two only when you need to narrow it:

```yaml
score_mode: best              # or final; defaults to best
required_final_score_max: "1.0"
```

`score_mode: best` suits evals with high per-point noise (for example, 30-question AIME); `final` suits "run to the end and pass at the end". `required_final_score_max` rejects obviously fabricated values.

## Add when the user specifies a scoring method

Reference scores are not limited to training scores. For each one, give the key name, threshold, weight, where it appears, and one log sample, and state the pass rule.

```yaml
score_components:
  - role: validation
    key: val-core/math_dapo/acc/mean@1
    required: "0.07"
    weight: "0.8"
    source: text_log
    log_sample: "step:200 - val-core/math_dapo/acc/mean@1:0.0402"
  - role: training_reward
    key: critic/rewards/mean
    required: "0.14"
    weight: "0.2"
    source: text_log
    log_sample: "step:200 - critic/rewards/mean:0.16"
pass_rule: "score = sum over components of weight times whether the component meets its threshold (1 if met, else 0); pass only when score is 1"
```

`role` determines the key names in `grader_spec.json`. Write them as a flat triple; **do not** generate `components` / `pass_rule` fields:

```json
{
  "metric_key": "val-core/math_dapo/acc/mean@1",
  "required_final_score": "0.07",
  "validation_weight": "0.8",
  "training_reward_key": "critic/rewards/mean",
  "required_training_reward": "0.14",
  "training_weight": "0.2"
}
```

`role: learning_rate` maps to `learning_rate_key` / `required_learning_rate` / `learning_rate_weight`. The eval-set side is always `validation_weight`.

Implement `pass_rule` exactly as the user wrote it. If the user did not provide `score_components`, do not apply this weighting.

## Add when multi-GPU or large shm is needed

```yaml
gpus: "0,1,2,3,4,5,6,7"
needs_compose: true
shm_size: "16gb"
```

With `needs_compose: true`, copy `tasks/reward/retool_length_hack/environment/docker-compose.yaml` verbatim (changing one line for `shm_size`). It only applies to the harbor backend; the native docker path derives `--gpus` itself from `NVIDIA_VISIBLE_DEVICES` in `environment.env` of `metadata.yaml`.

## Add when an asset needs on-site conversion

When no upstream repo exists and the asset can only be converted from downloaded assets (for example HF → Megatron torch_dist), give the conversion command and the arguments it depends on:

```yaml
assets:
  - kind: derive
    from: font-info_qwen3-4b-sft
    dest: font-info_qwen3-4b-sft_torch_dist
    command: tools/convert_hf_to_torch_dist.py
    fixed_args:
      WORLD_SIZE: "1"
      --rotary-base: "5000000"
    skip_without_cuda: true
    guard_in_launch_script: 'if [ ! -d "$DIST_MODEL_DIR" ]'
```

With `skip_without_cuda: true`, `prepare.sh` must probe for CUDA first and, if unavailable, print one line and skip. The harbor backend runs prepare as a build-time layer with no GPU, so the `guard_in_launch_script` guard does the work in the agent stage. If the output is not byte-for-byte reproducible, do not write `--expect` for it.
