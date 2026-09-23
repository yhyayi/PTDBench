window.PTDBENCH_INSTRUCTION_DATA = {
    issue: {
        title: 'RL training run diagnosis',
        paragraphs: [
            'Review the staged post-training experiment and determine whether it is healthy or unhealthy. Treat the workspace as a historical snapshot: inspect its source, effective configuration, launch scripts, data processing, reward logic, logs, tracked metrics, and frozen evaluation evidence. Do not assume that an unusual setting or implementation is causal without connecting it to observed behavior.',
            'Use held-out evaluation evidence together with rollout, truncation, runtime, optimization, and training signals. Training reward alone does not establish held-out health. A truncated log is not by itself proof of a defect.',
            'Do not launch a full training run, use the network, or modify the historical snapshot. The dataset, held-out set, model, training-step budget, and evaluation interval are fixed.'
        ],
        deliverablePath: '$PTDBENCH_RESULTS/agent_answer.md',
        deliverable: 'Write one concise UTF-8 diagnosis. State the verdict, identify the concrete root cause or causes, and explain the causal chain using specific workspace evidence. Distinguish root causes from symptoms. Do not propose or implement a fix, and do not write other deliverable files.'
    },
    solution: {
        title: 'Improve the staged experiment',
        paragraphs: [
            'Review the staged post-training experiment and determine whether it is healthy or unhealthy. Treat the workspace as a snapshot you may repair: inspect its source, effective configuration, launch scripts, data processing, reward logic, logs, tracked metrics, and frozen evaluation evidence. Do not assume that an unusual setting or implementation is causal without connecting it to observed behavior.',
            'Use held-out evaluation evidence together with rollout, truncation, runtime, optimization, and training signals. Training reward alone does not establish held-out health.'
        ]
    },
    tasks: {
        'reward_design/task_evaluator': {
            workspace: '/workspace',
            launch: 'examples/grpo_trainer/run_qwen2_5_1.5b_instruct_taco_fsdp.sh',
            effectiveRun: [
                'MODEL_PATH=/workspace/Qwen2.5-1.5B-Instruct',
                'TRAIN_FILE=/workspace/taco_verl/train.parquet',
                'TEST_FILE=/workspace/taco_verl/test.parquet',
                'LOG_DIR=$PTDBENCH_RESULTS/experiment_logs',
                'WANDB_MODE=offline',
                'WANDB_DIR=$PTDBENCH_RESULTS/experiment_logs',
                'WANDB_API_KEY=offline'
            ],
            strictLogs: true
        },
        'reward_design/task_function_call': {
            workspace: '/workspace',
            launch: 'examples/grpo_trainer/run_qwen2_5_1.5b_instruct_taco_fsdp.sh',
            effectiveRun: [
                'MODEL_PATH=/workspace/Qwen2.5-1.5B-Instruct',
                'TRAIN_FILE=/workspace/taco_verl/train.parquet',
                'TEST_FILE=/workspace/taco_verl/test.parquet',
                'LOG_DIR=$PTDBENCH_RESULTS/experiment_logs',
                'WANDB_MODE=offline',
                'WANDB_DIR=$PTDBENCH_RESULTS/experiment_logs',
                'WANDB_API_KEY=offline'
            ],
            strictLogs: true
        },
        'rlve_hyper/task_001': {
            workspace: '/workspace',
            launch: 'scripts/training/Qwen2.5-0.5B-Instruct/rlve.sh',
            launchNote: 'The script requires WANDB_PROJECT, RUN_NAME, and a dummy environment-list token. The locked environment list inside the script is used; do not replace it.',
            command: 'bash scripts/training/Qwen2.5-0.5B-Instruct/rlve.sh ptdbench solution_rerun _',
            effectiveRun: [
                'model /workspace/Qwen2.5-0.5B-Instruct',
                'Megatron dist checkpoint /workspace/Qwen2.5-0.5B-Instruct_torch_dist',
                'held-out eval data/HELD-OUT_ENVIRONMENTS/test_128.json',
                'LOG_DIR=$PTDBENCH_RESULTS/experiment_logs',
                'WANDB_MODE=offline',
                'WANDB_DIR=$PTDBENCH_RESULTS/experiment_logs'
            ]
        },
        'rlve_hyper/task_008': {
            workspace: '/workspace',
            launch: 'scripts/training/Qwen2.5-0.5B-Instruct/rlve.sh',
            issueNote: 'This run only trains 100 steps, numbered 0 through 99.',
            solutionNote: 'The training budget is fixed at 100 steps, numbered 0 through 99. Do not extend that budget.',
            launchNote: 'The script requires WANDB_PROJECT, RUN_NAME, and a dummy environment-list token. The locked environment list inside the script is used; do not replace it.',
            command: 'bash scripts/training/Qwen2.5-0.5B-Instruct/rlve.sh ptdbench solution_rerun _',
            effectiveRun: [
                'model /workspace/Qwen2.5-0.5B-Instruct',
                'Megatron dist checkpoint /workspace/Qwen2.5-0.5B-Instruct_torch_dist',
                'held-out eval data/HELD-OUT_ENVIRONMENTS/test_128.json',
                'LOG_DIR=$PTDBENCH_RESULTS/experiment_logs',
                'WANDB_MODE=offline',
                'WANDB_DIR=$PTDBENCH_RESULTS/experiment_logs',
                'training budget 100 steps (0-99)'
            ]
        },
        'rlve_hyper/task_012': {
            workspace: '/workspace',
            launch: 'scripts/training/Qwen2.5-0.5B-Instruct/rlve.sh',
            launchNote: 'The script requires WANDB_PROJECT, RUN_NAME, and a dummy environment-list token. The locked environment list inside the script is used; do not replace it.',
            command: 'bash scripts/training/Qwen2.5-0.5B-Instruct/rlve.sh ptdbench solution_rerun _',
            effectiveRun: [
                'model /workspace/Qwen2.5-0.5B-Instruct',
                'Megatron dist checkpoint /workspace/Qwen2.5-0.5B-Instruct_torch_dist',
                'held-out eval data/HELD-OUT_ENVIRONMENTS/test_128.json',
                'LOG_DIR=$PTDBENCH_RESULTS/experiment_logs',
                'WANDB_MODE=offline',
                'WANDB_DIR=$PTDBENCH_RESULTS/experiment_logs'
            ]
        },
        'rlve_hyper/task_019': {
            workspace: '/workspace',
            launch: 'scripts/training/Qwen2.5-0.5B-Instruct/rlve.sh',
            launchNote: 'The script requires WANDB_PROJECT, RUN_NAME, and a dummy environment-list token. The locked environment list inside the script is used; do not replace it.',
            command: 'bash scripts/training/Qwen2.5-0.5B-Instruct/rlve.sh ptdbench solution_rerun _',
            effectiveRun: [
                'model /workspace/Qwen2.5-0.5B-Instruct',
                'Megatron dist checkpoint /workspace/Qwen2.5-0.5B-Instruct_torch_dist',
                'held-out eval data/HELD-OUT_ENVIRONMENTS/test_128.json',
                'LOG_DIR=$PTDBENCH_RESULTS/experiment_logs',
                'WANDB_MODE=offline',
                'WANDB_DIR=$PTDBENCH_RESULTS/experiment_logs'
            ]
        },
        'verl_hyper/qwen_dapo_hparam/task_hparam_overlong_023': {
            workspace: '/workspace',
            launch: 'examples/grpo_trainer/run_qwen2_5_0_5b_instruct_dapo_math_fsdp.sh',
            effectiveRun: [
                'model /workspace/verl/Qwen2.5-0.5B-Instruct',
                'train data /workspace/DAPO-Math-17k_verl/train.parquet',
                'validation data /workspace/DAPO-Math-17k_verl/test.parquet',
                'LOG_DIR=$PTDBENCH_RESULTS/experiment_logs',
                'WANDB_MODE=offline',
                'WANDB_DIR=$PTDBENCH_RESULTS/experiment_logs'
            ]
        },
        'verl_implementation/data_format/task_agent_loop_022-llama-dapo-math': {
            workspace: '/workspace/verl',
            launch: 'examples/grpo_trainer/run_llama3_2_1b_instruct_dapo_math_fsdp.sh',
            effectiveRun: [
                'LOG_DIR=$PTDBENCH_RESULTS/experiment_logs',
                'WANDB_DIR=$PTDBENCH_RESULTS/experiment_logs',
                'WANDB_MODE=offline',
                'WANDB_API_KEY=offline',
                'MODEL_PATH=/workspace/verl/Llama-3.2-1B-instruct',
                'TRAIN_FILE=/workspace/DAPO-Math-17k_verl/train.parquet',
                'TEST_FILE=/workspace/DAPO-Math-17k_verl/test.parquet'
            ]
        },
        'verl_implementation/llama_dapo_implementation/task_monkey_patch_011': {
            workspace: '/workspace',
            launch: 'examples/grpo_trainer/run_llama3_2_1b_instruct_dapo_math_fsdp.sh',
            effectiveRun: [
                'MODEL_PATH=/workspace/verl/Llama-3.2-1B-instruct',
                'TRAIN_FILE=/workspace/DAPO-Math-17k_verl/train.parquet',
                'TEST_FILE=/workspace/DAPO-Math-17k_verl/test.parquet',
                'LOG_DIR=$PTDBENCH_RESULTS/experiment_logs',
                'WANDB_MODE=offline',
                'WANDB_DIR=$PTDBENCH_RESULTS/experiment_logs',
                'WANDB_API_KEY=offline'
            ],
            strictLogs: true
        },
        'verl_implementation/verl_reward_implementation/task_engine_base_025': {
            workspace: '/workspace',
            launch: 'examples/grpo_trainer/run_qwen2_5_0.5b_instruct_fsdp.sh',
            effectiveRun: [
                'MODEL_PATH=/workspace/verl/Qwen2.5-0.5B-Instruct',
                'TRAIN_FILE=/workspace/gsm8k_verl/train.parquet',
                'TEST_FILE=/workspace/gsm8k_verl/test.parquet',
                'LOG_DIR=$PTDBENCH_RESULTS/experiment_logs',
                'WANDB_MODE=offline',
                'WANDB_DIR=$PTDBENCH_RESULTS/experiment_logs',
                'WANDB_API_KEY=offline'
            ],
            strictLogs: true
        }
    }
};
