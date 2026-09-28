from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from .types import (
    DEFAULT_DATASETS_DIR,
    DEFAULT_MODELS_DIR,
    CommandResult,
    EvaluationPaths,
    EvaluationResult,
    TaskPackage,
    _normalize_plain_url,
)


def harbor_task_name(task_id: str, evaluation_level: str) -> str:
    """Harbor task 名；同时用于把 trial 结果映射回 PTDBench task。"""
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", task_id).strip("-.") or "task"
    return f"ptdbench/{safe_id}-{evaluation_level}"


@dataclass(frozen=True)
class HarborEvaluationConfig:
    """PTDBench 转换参数；Harbor 自身参数统一放在 harbor_args。

    Harbor 直接用预构建镜像（task.toml 的 `docker_image`），不再逐任务构建，
    所以 verifier 固定与 agent 共用同一个容器（shared 模式）。prepare.sh 在
    `harbor run` 之前由 PTDBench 在同一个镜像里跑完，产物以只读挂载交给 Harbor。
    """

    evaluation_level: str = "solution"
    judge_model: Optional[str] = None
    agent_network: str = "public"
    verifier_network: str = "public"
    temp_dir: Optional[Path] = None
    # prepare 阶段用的 HF 镜像站；只影响 prepare，agent / verifier 是离线的。
    huggingface_mirror: Optional[str] = None
    # 与 EvaluationConfig 同一套下载缓存，留空落到仓库根的 models/ 和 datasets/。
    models_dir: Optional[Path] = None
    datasets_dir: Optional[Path] = None
    harbor_command: Sequence[str] = field(default_factory=lambda: ("harbor",))
    harbor_args: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.evaluation_level not in {"issue", "solution"}:
            raise ValueError("evaluation_level must be 'issue' or 'solution'")
        network_modes = {"no-network", "public", "allowlist"}
        if self.agent_network not in network_modes:
            raise ValueError(f"unsupported agent_network: {self.agent_network}")
        if self.verifier_network not in network_modes:
            raise ValueError(f"unsupported verifier_network: {self.verifier_network}")
        if isinstance(self.harbor_command, str) or not self.harbor_command:
            raise ValueError("harbor_command must be a non-empty sequence")
        if any(not isinstance(arg, str) or not arg for arg in self.harbor_command):
            raise ValueError("harbor_command must contain non-empty strings")
        if isinstance(self.harbor_args, str):
            raise ValueError("harbor_args must be a sequence of argv tokens")
        if any(not isinstance(arg, str) or not arg for arg in self.harbor_args):
            raise ValueError("harbor_args must contain non-empty strings")
        if isinstance(self.temp_dir, str) and not self.temp_dir.strip():
            raise ValueError("temp_dir must not be empty")
        if self.huggingface_mirror is not None:
            object.__setattr__(
                self, "huggingface_mirror", _normalize_plain_url(self.huggingface_mirror)
            )

        if self.temp_dir is not None:
            object.__setattr__(self, "temp_dir", Path(self.temp_dir).expanduser())
        for name, fallback in (
            ("models_dir", DEFAULT_MODELS_DIR),
            ("datasets_dir", DEFAULT_DATASETS_DIR),
        ):
            given = getattr(self, name)
            resolved = (Path(given) if given is not None else fallback).expanduser().resolve()
            object.__setattr__(self, name, resolved)
        object.__setattr__(self, "harbor_command", tuple(self.harbor_command))
        object.__setattr__(self, "harbor_args", tuple(self.harbor_args))


@dataclass(frozen=True)
class HarborTaskUnit:
    """批次中的一个待评测单元：一个 task 的一个 evaluation。"""

    package: TaskPackage
    evaluation: EvaluationPaths
    workspace_before_hash: str
    # prepare 阶段跑完之后宿主机上的两个目录。workspace 是 prepare 写过的那份副本
    # （权重可能就下在里面），物化时整份放进 environment/，由 Harbor 在容器启动后
    # 上传到 /workspace；prepared 以只读挂到 /ptdbench/prepared。都为 None 表示还
    # 没 staging，这时直接用任务自带的 workspace、不挂 prepared。
    workspace: Optional[Path] = None
    prepared: Optional[Path] = None

    @property
    def harbor_task_name(self) -> str:
        """写进 task.toml 的 Harbor task 名，也是 trial 的回溯键。"""
        return harbor_task_name(self.package.task_id, self.evaluation.level)

    @property
    def dir_name(self) -> str:
        """批次目录下的子目录名；Harbor task 名含 '/'，不能直接做目录。"""
        return self.harbor_task_name.split("/", 1)[1]


@dataclass(frozen=True)
class PreparedHarborTask:
    """在线生成的一批临时 Harbor task。"""

    root: Path
    tasks_dir: Path
    units: Sequence[HarborTaskUnit] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "units", tuple(self.units))

    def task_dir(self, unit: HarborTaskUnit) -> Path:
        return self.tasks_dir / unit.dir_name

    @property
    def units_by_task_name(self) -> Dict[str, HarborTaskUnit]:
        return {unit.harbor_task_name: unit for unit in self.units}


@dataclass(frozen=True)
class HarborRun:
    """Harbor CLI 的进程结果及其确定的 job 目录。"""

    command: CommandResult
    job_dir: Path


@dataclass(frozen=True)
class HarborEvaluationReport:
    """一个 Harbor job 中所有 trial 的 PTDBench 结果。"""

    task_ids: Sequence[str]
    evaluation_level: str
    trials: Sequence[EvaluationResult]
    report_dir: Path
    command: CommandResult

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_ids", tuple(self.task_ids))
        object.__setattr__(self, "trials", tuple(self.trials))

    @property
    def completed_task_ids(self) -> tuple[str, ...]:
        completed = {trial.task_id for trial in self.trials}
        return tuple(task_id for task_id in self.task_ids if task_id in completed)

    @property
    def missing_task_ids(self) -> tuple[str, ...]:
        completed = {trial.task_id for trial in self.trials}
        return tuple(task_id for task_id in self.task_ids if task_id not in completed)

    @property
    def passed(self) -> bool:
        return (
            bool(self.task_ids)
            and not self.missing_task_ids
            and all(trial.passed for trial in self.trials)
        )

    @property
    def score(self) -> Optional[float]:
        scores = [trial.score for trial in self.trials if trial.score is not None]
        return sum(scores) / len(scores) if scores else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_ids": list(self.task_ids),
            "evaluation_level": self.evaluation_level,
            "passed": self.passed,
            "score": self.score,
            "n_tasks": len(self.task_ids),
            "n_completed_tasks": len(self.completed_task_ids),
            "missing_task_ids": list(self.missing_task_ids),
            "n_trials": len(self.trials),
            "n_passed": sum(trial.passed for trial in self.trials),
            "report_dir": str(self.report_dir),
            "command": self.command.to_dict(),
            "trials": [trial.to_dict() for trial in self.trials],
        }


__all__ = [
    "HarborEvaluationConfig",
    "HarborEvaluationReport",
    "HarborRun",
    "HarborTaskUnit",
    "PreparedHarborTask",
    "harbor_task_name",
]
