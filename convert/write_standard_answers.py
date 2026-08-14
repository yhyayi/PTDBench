#!/usr/bin/env python3
"""Write English verifier-only attribution answers for 58 selected tasks."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple


TaskKey = Tuple[str, str]


@dataclass(frozen=True)
class Answer:
    group: str
    verdict: str
    attribution: str
    requirements: Tuple[str, ...]


def unhealthy(group: str, attribution: str, *requirements: str) -> Answer:
    return Answer(group, "unhealthy", attribution, requirements)


def healthy(group: str, reason: str) -> Answer:
    return Answer(
        group,
        "healthy",
        "none",
        (
            "Explicitly classify the experiment as healthy.",
            reason,
        ),
    )


ANSWERS: Dict[TaskKey, Answer] = {
    ("qwen_dapo_hparam", "task_hparam_overlong_023"): unhealthy(
        "A-core-verl-hparam",
        "MAX_RESPONSE_LENGTH=256; overlong fields behind the disabled enable gate are distractors.",
        "Name MAX_RESPONSE_LENGTH=256.",
        "Exclude gated OVERLONG_BUFFER_LEN and OVERLONG_PENALTY_FACTOR fields as inactive distractors.",
    ),
    ("qwen_dapo_hparam", "task_hparam_seqlen_micro_014"): unhealthy(
        "A-core-verl-hparam",
        "The MAX_PROMPT_LENGTH / MAX_RESPONSE_LENGTH length family.",
        "Name MAX_PROMPT_LENGTH and MAX_RESPONSE_LENGTH.",
    ),
    ("qwen_dapo_hparam", "task_hparam_clip_grpo_022"): unhealthy(
        "A-challenge-verl-hparam",
        "ROLLOUT_N: 8 -> 2.",
        "Name ROLLOUT_N and state 8 -> 2.",
    ),
    ("qwen_dapo_hparam", "task_hparam_rollout_temp_n_019"): unhealthy(
        "A-challenge-verl-hparam",
        "ROLLOUT_N=1 plus temperature/top-p and response length.",
        "Name ROLLOUT_N=1, temperature/top-p, and response length.",
    ),
    ("qwen_dapo_hparam", "task_hparam_rollout_len_sample_020"): unhealthy(
        "A-challenge-verl-hparam",
        "MAX_RESPONSE_LENGTH=32 plus ROLLOUT_N=1.",
        "Name MAX_RESPONSE_LENGTH=32 and ROLLOUT_N=1.",
    ),
    ("qwen_dapo_hparam", "task_hparam_batch_clip_015"): healthy(
        "A-healthy-negative",
        "The batch/clip differences create no effective semantic fault.",
    ),
    ("llama_dapo_implementation", "task_monkey_patch_011"): unhealthy(
        "B-core-verl-implementation",
        "monkey_patch.py: repeat_kv tensor layout or reduced apply_monkey_patch behavior.",
        "Name monkey_patch.py and either repeat_kv layout or missing apply_monkey_patch behavior.",
    ),
    ("data_format", "task_agent_loop_022-llama-dapo-math"): unhealthy(
        "B-core-data-format",
        "DAPO-Math data format plus the agent-loop implementation.",
        "Identify both the data-format contract and the agent-loop implementation.",
    ),
    ("verl_reward_implementation", "task_engine_base_025"): unhealthy(
        "B-core-reward-engine",
        "Reward-engine implementation in verl/workers/engine/base.py; 26 methods were rewritten.",
        "Name verl/workers/engine/base.py and the broad 26-method rewrite.",
    ),
    ("llama_dapo_implementation", "task_core_algos_013"): unhealthy(
        "B-challenge-verl-implementation",
        "Global token-mean normalization in agg_loss.",
        "Name agg_loss and token-mean global normalization.",
    ),
    ("llama_dapo_implementation", "task_kl_reward_adv_chain_032"): unhealthy(
        "B-challenge-verl-implementation",
        "The advantage/reward computation chain, not KL alone.",
        "Identify the linked advantage and reward computation path.",
    ),
    ("data_format", "task_agent_loop_022-qwen-math-dapo-math"): unhealthy(
        "B-challenge-data-format",
        "Qwen-Math data format plus the agent-loop implementation.",
        "Identify both the Qwen-Math data-format contract and agent-loop implementation.",
    ),
    (
        "data_format",
        "task_alt_adv_rloo_reinforce_039-qwen-math-dapo-math",
    ): unhealthy(
        "B-challenge-data-format",
        "Data format plus the RLOO/REINFORCE++ advantage implementation.",
        "Identify both the data-format contract and RLOO/REINFORCE++ advantage path.",
    ),
    ("data_format", "task_exp_dapo_rm_007-llama-dapo-math"): unhealthy(
        "B-challenge-data-format",
        "Data format in reward-manager answer handling.",
        "Identify the data-format contract and reward manager.",
    ),
    ("data_format", "task_core_algos_013-llama-dapo-math"): unhealthy(
        "B-challenge-data-format",
        "Data format plus agg_loss token-mean global normalization.",
        "Identify both the data-format contract and agg_loss token-mean normalization.",
    ),
    ("qwen_dapo_implementation", "task_ray_trainer_015"): healthy(
        "B-healthy-negative",
        "The ray_trainer.py difference is semantically equivalent.",
    ),
    (
        "qwen_dapo_implementation",
        "task_reward_adv_grpo_core_031",
    ): healthy(
        "B-healthy-negative",
        "The GRPO-core difference is semantically equivalent.",
    ),
    ("qwen_dapo_implementation", "task_rl_dataset_026"): healthy(
        "B-healthy-negative",
        "The rl_dataset.py difference is semantically equivalent.",
    ),
}


RLVE_UNHEALTHY = {
    "task_001": "attention-dropout=0.5.",
    "task_005": "eval-temperature=2, an evaluation-side fault.",
    "task_008": "lr=1e-9.",
    "task_009": "lr=1e-8.",
    "task_010": "eval-max-response-len=20, an evaluation-side fault.",
    "task_012": "hidden-dropout=0.5.",
    "task_016": "attention-dropout=0.7.",
    "task_019": "lr-warmup-iters=2000 plus lr-decay-iters=10000.",
    "task_002": "kl-coef=10 plus lr=1e-4.",
    "task_003": "lr=1e-5.",
    "task_015": "attention-dropout=0.3.",
    "task_018": "rollout-temperature=0.",
    "task_021": "entropy-coef=0.1.",
}
for task_id, attribution in RLVE_UNHEALTHY.items():
    ANSWERS[("rlve_hyper", task_id)] = unhealthy(
        "C-rlve-hparam",
        attribution,
        f"Name the exact attribution: {attribution}",
    )
ANSWERS[("rlve_hyper", "task_006")] = healthy(
    "C-rlve-healthy-negative",
    "It has zero deviation from the majority-vote baseline.",
)
ANSWERS[("rlve_hyper", "task_007")] = healthy(
    "C-rlve-healthy-negative",
    "attention-dropout=0.1 is an inert change.",
)


REWARD_UNHEALTHY = {
    "reward_difference_constraint_system_034": "core",
    "reward_integer_programming_029": "core",
    "reward_land_acquisition_008": "core",
    "reward_max_different_group_pair_division_023": "core",
    "reward_min_cost_reducing_lnds_020": "core",
    "reward_min_pair_sum_multiplication_permutation_007": "core",
    "reward_new_nim_game_021": "core",
    "reward_polynomial_factorization_035": "core",
    "reward_prefix_product_mod_distinct_permutation_011": "core",
    "reward_prefix_sum_mod_distinct_permutation_010": "core",
    "reward_quadratic_function_segmentation_019": "core",
    "reward_sat_031": "core",
    "reward_set_splitting_012": "core",
    "reward_topological_sort_minimal_lexicographical_order_001": "core",
    "reward_visible_line_038": "core",
    "reward_bez_minimalist_security_002": "challenge",
    "reward_campfire_party_016": "challenge",
    "reward_difference_constraint_system_dag_003": "challenge",
    "reward_dyn_dynamite_017": "challenge",
    "reward_maf_mafia_004": "challenge",
    "reward_min_cost_tree_coverage_014": "challenge",
    "reward_protecting_flowers_006": "challenge",
    "reward_two_sat_030": "challenge",
}
for task_id, level in REWARD_UNHEALTHY.items():
    slug = task_id.removeprefix("reward_")
    ANSWERS[("reward_design", task_id)] = unhealthy(
        f"D-{level}-rlve-reward-design",
        f"The reward design or reward function for the {slug} environment.",
        f"Name the {slug} environment and identify its reward design or reward function.",
    )
for task_id in ("reward_tetris_attack_025", "reward_hungry_rabbit_036"):
    slug = task_id.removeprefix("reward_")
    ANSWERS[("reward_design", task_id)] = healthy(
        "D-healthy-negative",
        f"The current reward implementation for {slug} must be treated as healthy.",
    )


def render(task_id: str, answer: Answer) -> str:
    lines = [
        "STANDARD_ANSWER_V1",
        f"TASK_ID: {task_id}",
        f"BENCHMARK_GROUP: {answer.group}",
        f"EXPECTED_VERDICT: {answer.verdict}",
        f"EXPECTED_ATTRIBUTION: {answer.attribution}",
        "",
        "REQUIRED_ALL:",
        *(f"- {item}" for item in answer.requirements),
        "",
        "GLOBAL_JUDGE_POLICY:",
        "- The score must be exactly 0 or 1.",
        "- Agent A must explicitly classify the run as healthy or unhealthy.",
        "- An unhealthy answer may provide only one primary attribution.",
        "- A gold parameter bundle or computation chain counts as one attribution.",
        "- Broad labels without the required concrete attribution are insufficient.",
        "- Every REQUIRED_ALL item must be satisfied.",
        "- A verdict mismatch forces score 0.",
        "- Equivalent technical wording in any language is acceptable.",
        "- task.yaml may clarify targets and aliases but may not relax REQUIRED_ALL.",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve()
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-root", type=Path, default=here.parents[1] / "tasks")
    return parser.parse_args()


def write_one(task_root: Path) -> Path:
    answer = ANSWERS[(task_root.parent.name, task_root.name)]
    path = task_root / "oracle/standard_answer.txt"
    path.write_text(render(task_root.name, answer), encoding="utf-8")
    return path


def main() -> int:
    tasks_root = parse_args().tasks_root.resolve()
    actual = {
        (path.parent.parent.name, path.parent.name)
        for path in tasks_root.glob("*/*/metadata.yaml")
    }
    if actual != set(ANSWERS):
        raise ValueError(
            f"answer map mismatch: missing={sorted(actual - set(ANSWERS))}, "
            f"extra={sorted(set(ANSWERS) - actual)}"
        )
    for family, task_id in sorted(actual):
        write_one(tasks_root / family / task_id)
    healthy_count = sum(item.verdict == "healthy" for item in ANSWERS.values())
    print(
        f"wrote {len(ANSWERS)} English standard answers "
        f"({healthy_count} healthy, {len(ANSWERS) - healthy_count} unhealthy)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
