from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from ..executor import CommandResult, DockerExecutor, LocalExecutor
from ..filesystem import write_text
from ..types import AgentSpec


@dataclass(frozen=True)
class AgentExecutionContext:
    # agent 运行时只看这个上下文，不直接依赖 evaluator 内部结构。
    case_id: str
    instruction: str
    metadata: Dict[str, Any]
    timeout_s: Optional[int]
    local_workspace: Path
    local_visible_case: Path
    local_results: Path
    local_env: Dict[str, str] = field(default_factory=dict)
    container_workspace: str = "/workspace"
    container_visible_case: str = "/case"
    container_results: str = "/results"
    container_env: Dict[str, str] = field(default_factory=dict)


class AgentRunner(ABC):
    # 每个具体 agent 都实现这个接口；evaluator 只依赖这里。
    def __init__(self, spec: AgentSpec, default_backend: str = "local"):
        backend = spec.backend or default_backend
        if backend not in {"docker", "local"}:
            raise ValueError("agent backend must be 'docker' or 'local'")
        self.spec = spec
        self.backend = backend

    @property
    def kind(self) -> str:
        return self.spec.kind

    @property
    def needs_environment_image(self) -> bool:
        return self.backend == "docker"

    def _write_agent_trace(self, results_dir: Path, payload: Dict[str, Any]) -> Path:
        # 统一把 agent 的最终轨迹收敛成一个 JSON 文件，减少 results 里的碎片。
        path = results_dir / "agent_trace.json"
        write_text(path, json.dumps(payload, indent=2, ensure_ascii=False))
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
    def run(
        self,
        context: AgentExecutionContext,
        *,
        local: LocalExecutor,
        docker: DockerExecutor,
        image_tag: Optional[str],
    ) -> CommandResult:
        # 具体 agent 负责把自己的 spec 和 context 翻译成一次可执行调用。
        raise NotImplementedError
