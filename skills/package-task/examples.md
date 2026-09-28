# 用户输入示例

打包时按此结构一次给齐。缺一项就停。数字、commit 与 hash 需换成该次训练的真实值。

打包时阅读 buggy commit 与 baseline commit 的代码和启动脚本 diff，把根因写成一句话，并写上这里给出的指标现象。

## 只有评测分

```yaml
source_workspace: /home/test/test06/lyh/verl
output_dir: /home/test/test1708/data_call/tasks/verl_implementation/llama_dapo_implementation/task_example_001
task_id: task_example_001
title: "llama_dapo_implementation implementation diagnosis: task_example_001"
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
      启动脚本 :30-38 按顺序找 /models/… → /workspace/verl/Llama-3.2-1B-instruct，
      第一个候选没人挂，必须正好落在第二个上。目录名是小写 i。
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

按上述 diff 写出的 `failure_reason` 会类似：`apply_monkey_patch 不再安装 flash-attention hook，val-core 最终约 0.0402`。具体措辞以过滤后的 diff 为准。

`assets` 里每一条落成 `environment/prepare.sh` 的一次 `ptdbench_fetch_hf` / `ptdbench_dataset` / `ptdbench_fetch_url` 调用，`dest` 前面补 `$PTDBENCH_WORKSPACE/`。`why_this_path` 写成脚本里那条注释——路径错了要到训练中途才炸。
## 未写评分方式

上面的示例到 `required_final_score` 为止，就是默认评分：分数全部看评测集键。按 `score_mode` 取该键的最高有限值（默认 `best`）或 `final_step` 那一点（`final`），达到 `required_final_score` 则 score 为 1，否则为 0。

需要收窄时再给这两项：

```yaml
score_mode: best              # 或 final；不写按 best
required_final_score_max: "1.0"
```

`score_mode: best` 适合单点噪声大的评测（例如 30 道题的 AIME）；`final` 适合要求「跑到终点且终点达标」。`required_final_score_max` 用来挡掉明显编造的值。

## 用户指定评分方式时追加

参考分不限于训练分。每一项给出键名、门槛、权重、出现位置和一条 log 样例，并写明及格规则。

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
pass_rule: "score = 各项权重乘以该项是否达到门槛（达到为 1，否则为 0）之和；score 为 1 才算通过"
```

`role` 决定 `grader_spec.json` 里的键名，用扁平三件套写，**不要**生成 `components` / `pass_rule` 字段：

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

`role: learning_rate` 对应 `learning_rate_key` / `required_learning_rate` / `learning_rate_weight`（见 `tasks/rlve_hyper/task_008`）。评测集那一侧一律是 `validation_weight`。

`pass_rule` 按用户原文实现。用户没有写 `score_components` 时，不要套用这段加权。

## 需要多卡或大 shm 时追加

```yaml
gpus: "0,1,2,3,4,5,6,7"
needs_compose: true
shm_size: "16gb"
```

`needs_compose: true` 时把 `tasks/slime_retool/environment/docker-compose.yaml` 照抄过去（按 `shm_size` 改一处）。它只对 harbor 后端生效，原生 docker 路径从 `metadata.environment.env` 的 `NVIDIA_VISIBLE_DEVICES` 自己翻译 `--gpus`。

## 资产需要现场转换时追加

上游没有对应仓库、只能由已下载的资产转出来时（例如 HF → Megatron torch_dist），给出转换命令与它依赖的参数：

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

`skip_without_cuda: true` 时 `prepare.sh` 要先探测 CUDA，不可用就打印一行跳过——harbor 后端把 prepare 跑成 build-time 层，那里没有 GPU，由 `guard_in_launch_script` 那个守卫在 agent 阶段补做。产物不逐字节可复现时，不要给它写 `--expect`。
