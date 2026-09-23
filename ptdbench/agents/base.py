from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from ..core.types import AgentSpec, CommandResult
from ..runtimes import (
    EnvironmentArtifact,
    SandboxInstance,
    SandboxRequest,
    SandboxRuntime,
)
from ..utils.filesystem import write_text


@dataclass(frozen=True)
class AgentExecutionContext:
    # agent 运行时只看这个上下文，不直接依赖 evaluator 或具体 runtime。
    task_id: str
    instruction: str
    metadata: Dict[str, Any]
    timeout_s: Optional[int]
    task_root: Path
    workspace: Path
    visible_task: Path
    results: Path
    evaluation_level: str = "solution"
    log_level: str = "quiet"
    debian_mirror: Optional[str] = None
    # 统一镜像相关：要激活哪套 conda 环境。
    conda_env: Optional[str] = None
    # 任务的 environment/ 目录和 prepare 阶段的产物目录，都以只读挂给 agent。
    # agent 看得到 requirements.txt 和 prepare.sh 是有意的：它要改的是训练代码，
    # 而"环境里装了什么"是它做判断的前提，不是答案的一部分。
    environment_dir: Optional[Path] = None
    prepared_dir: Optional[Path] = None
    # 两个下载缓存根，只读挂给 agent。看得到是为了排查时对得上账——交付路径仍然是
    # prepare 用 --dest 写进 workspace 的那一份，任务脚本不该直接引用这两个目录。
    models_dir: Optional[Path] = None
    datasets_dir: Optional[Path] = None
    # metadata.environment.env：逐任务不同的变量（如 CUDA_VISIBLE_DEVICES）。
    task_env: Dict[str, str] = field(default_factory=dict)

    def sandbox_request(
        self,
        runtime: SandboxRuntime,
        *,
        env: Optional[Dict[str, str]] = None,
        network: str = "bridge",
        workdir: str = "workspace",
    ) -> SandboxRequest:
        """为 agent 创建统一的运行请求，路径细节交给 runtime 解析。"""
        return SandboxRequest(
            stage="agent",
            task_id=self.task_id,
            host_task_root=self.task_root,
            host_visible_task=self.visible_task,
            host_workspace=self.workspace,
            host_results=self.results,
            evaluation_level=self.evaluation_level,
            env=env or {},
            network=network,
            timeout_s=self.timeout_s,
            workdir=workdir,
            conda_env=self.conda_env,
            host_environment=self.environment_dir,
            host_prepared=self.prepared_dir,
            host_models=self.models_dir,
            host_datasets=self.datasets_dir,
            task_env=self.task_env,
        )


class AgentRunner(ABC):
    # 每个具体 agent 都实现这个接口；evaluator 只依赖这里。
    def __init__(self, spec: AgentSpec):
        self.spec = spec

    @property
    def kind(self) -> str:
        return self.spec.kind

    def _write_agent_trace(self, results_dir: Path, payload: Dict[str, Any]) -> Path:
        # 统一把 agent 的最终轨迹收敛成一个 JSON 文件，减少 results 里的碎片。
        path = results_dir / "agent_trace.json"
        write_text(path, json.dumps(payload, indent=2, ensure_ascii=False))
        return path

    def _write_agent_trace_jsonl(
        self,
        results_dir: Path,
        messages: Sequence[Dict[str, str]],
    ) -> Path:
        """按 OpenAI message 风格逐行保存完整 agent 轨迹。"""
        lines = []
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if not isinstance(role, str) or not isinstance(content, str):
                raise ValueError("trace message needs string role and content")
            lines.append(json.dumps({"role": role, "content": content}, ensure_ascii=False))
        path = results_dir / "agent_trace.jsonl"
        write_text(path, "\n".join(lines) + "\n")
        return path

    @staticmethod
    def _redact_env(env: Dict[str, str]) -> Dict[str, str]:
        redacted: Dict[str, str] = {}
        for key, value in env.items():
            if any(token in key.upper() for token in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
                redacted[key] = "<redacted>"
            else:
                redacted[key] = value
        return redacted

    @abstractmethod
    def setup(
        self,
        context: AgentExecutionContext,
        *,
        runtime: SandboxRuntime,
        sandbox: SandboxInstance,
    ) -> CommandResult:
        """Install/prep this agent in the already-created sandbox."""
        raise NotImplementedError

    @abstractmethod
    def run(
        self,
        context: AgentExecutionContext,
        *,
        runtime: SandboxRuntime,
        artifact: Optional[EnvironmentArtifact],
        sandbox: Optional[SandboxInstance] = None,
    ) -> CommandResult:
        # 具体 agent 只负责把 spec 翻译成命令；执行平台由 runtime 提供。
        raise NotImplementedError
