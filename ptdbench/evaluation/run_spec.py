"""solution 层的系统重跑（rerun）协议。

新的 solution 层协议：agent 只在 workspace 里修代码、提交 report.md，
**不启动训练**；harness 在 verifier 之前用 agent 留下的 workspace 重跑一遍
实验，然后由任务自己的判分脚本读新日志里的关键 metric。

这一份是协议的机器可读部分。launch 脚本的默认值和协议要求的预算经常不一致
（search-r1 脚本默认 `--num-rollout 200`，协议要求 100；retool 默认 3000，
要求 70），预算覆盖只写在 instruction.md 的散文里时，harness 没法替 agent
把它跑对。所以每个 training 任务在
`evaluations/solution/verifier/resources/run_spec.json` 里固化一份：
跑哪条命令、带哪些环境变量、多久超时、日志落在哪。

没有 run_spec.json 的 solution 任务（verifier 自己驱动训练的
critic_value_head_random_init、纯协议修复的 aime25_single_sample_variance）
同样适用新协议的 agent 侧约束——修代码、交 report.md、不跑训练——只是
harness 不额外跑一遍实验。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

RUN_SPEC_FILENAME = "run_spec.json"
RUN_SPEC_SCHEMA_VERSION = 1
# env 值里允许引用的 harness 变量。写成 $PTDBENCH_RESULTS 是为了跨后端可移植：
# docker 容器里它是 /results，Harbor 的 verifier 容器里是 /logs/artifacts，
# local 模式是宿主机 results 目录。展开发生在执行侧，spec 里不写死。
_EXPANDABLE_PREFIX = "$PTDBENCH_"
REPORT_FILENAME = "report.md"


class RunSpecError(ValueError):
    """run_spec.json 存在但不合法：这是任务定义的错，得原样报给任务作者。"""


@dataclass(frozen=True)
class RunSpec:
    """rerun 阶段执行一次实验所需的全部信息。

    全部来自任务仓库（agent 不可见），没有 agent 能影响的部分。
    """

    task_id: str
    # 在 workspace 根执行的 shell 命令，含协议要求的显式参数覆盖。
    command: str
    # 传给命令的环境变量。值里可以引用 $PTDBENCH_RESULTS /
    # $PTDBENCH_WORKSPACE，由执行侧展开成当前后端的真实路径。
    env: Dict[str, str]
    # 命令的墙钟预算（秒）。None 时沿用 EvaluationConfig.timeout_s。
    timeout_s: Optional[int]
    # 实验日志相对 $PTDBENCH_RESULTS 的根。rerun 前会清掉这些目录：
    # 判分只认系统重跑的新日志，agent 阶段就算自己跑过也白跑。
    log_roots: List[str]

    def expand_env(self, values: Dict[str, str]) -> Dict[str, str]:
        """把 env 值里的 $PTDBENCH_* 引用换成当前后端的真实路径。

        values 是本后端 / 本次请求的 PTDBENCH_* 解析结果（键不带 $）。
        引用了不存在的变量时原样保留——比抛错好：报错会把一个 run_spec 的
        小笔误放大成整个评测的 orchestration failure。
        """
        expanded: Dict[str, str] = {}
        for key, value in self.env.items():
            if _EXPANDABLE_PREFIX not in value:
                expanded[key] = value
                continue
            for name, resolved in values.items():
                value = value.replace(
                    f"${name}", resolved if resolved else f"${name}"
                )
            expanded[key] = value
        return expanded


def load_run_spec(evaluation_root: Path, task_id: str) -> Optional[RunSpec]:
    """加载任务的 run_spec.json；任务没有就返回 None（不启用系统重跑）。

    存在但不合法必须当场报错：rerun 阶段跑一半才发现 spec 写坏了，报出来
    像是 harness 的错，其实该找任务作者。
    """
    path = evaluation_root / "verifier" / "resources" / RUN_SPEC_FILENAME
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunSpecError(f"{path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RunSpecError(f"{path}: top level must be an object")
    version = payload.get("schema_version")
    if version != RUN_SPEC_SCHEMA_VERSION:
        raise RunSpecError(
            f"{path}: schema_version must be {RUN_SPEC_SCHEMA_VERSION}, "
            f"got {version!r}"
        )
    command = payload.get("command")
    if not isinstance(command, str) or not command.strip():
        raise RunSpecError(f"{path}: command must be a non-empty shell string")
    env = payload.get("env", {})
    if not isinstance(env, dict):
        raise RunSpecError(f"{path}: env must be an object")
    for key, value in env.items():
        if not isinstance(key, str) or not key:
            raise RunSpecError(f"{path}: env keys must be non-empty strings")
        if not isinstance(value, (str, int, float, bool)):
            raise RunSpecError(
                f"{path}: env[{key!r}] must be a scalar, got {type(value).__name__}"
            )
        if key.startswith("PTDBENCH_"):
            raise RunSpecError(
                f"{path}: env must not declare {key}; the harness injects "
                "PTDBENCH_* variables itself"
            )
    log_roots = payload.get("log_roots", ["experiment_logs"])
    if not isinstance(log_roots, list) or not all(
        isinstance(item, str) and item for item in log_roots
    ):
        raise RunSpecError(
            f"{path}: log_roots must be a list of non-empty strings"
        )
    timeout_raw = payload.get("timeout_s")
    if timeout_raw is not None and (
        not isinstance(timeout_raw, (int, float))
        or isinstance(timeout_raw, bool)
        or timeout_raw <= 0
    ):
        raise RunSpecError(f"{path}: timeout_s must be a positive number")
    return RunSpec(
        task_id=task_id,
        command=command,
        env={key: str(value) for key, value in env.items()},
        timeout_s=None if timeout_raw is None else int(timeout_raw),
        log_roots=list(log_roots),
    )
