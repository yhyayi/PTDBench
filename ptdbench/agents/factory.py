from __future__ import annotations

from typing import Callable, Dict, Optional, Type

from ..types import AgentSpec
from .base import AgentRunner

_REGISTRY: Dict[str, Type[AgentRunner]] = {}


def register_agent_runner(kind: str) -> Callable[[Type[AgentRunner]], Type[AgentRunner]]:
    # 新 agent 只需要注册自己，不需要改 evaluator。
    def decorator(cls: Type[AgentRunner]) -> Type[AgentRunner]:
        _REGISTRY[kind] = cls
        return cls

    return decorator


def resolve_agent_spec(agent: Optional[AgentSpec], legacy_cmd: Optional[str]) -> Optional[AgentSpec]:
    # 先兼容旧的 --agent-cmd；如果新式 AgentSpec 已经给了，就直接用新的。
    if agent is not None:
        return agent
    if legacy_cmd:
        return AgentSpec(kind="shell", command=legacy_cmd)
    return None


def create_agent_runner(spec: AgentSpec, default_backend: str) -> AgentRunner:
    runner_cls = _REGISTRY.get(spec.kind)
    if runner_cls is None:
        raise ValueError(f"unsupported agent kind: {spec.kind}")
    return runner_cls(spec, default_backend=default_backend)


# 导入 installed agents 以完成注册。
from . import installed as _installed  # noqa: E402,F401
