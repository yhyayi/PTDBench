from __future__ import annotations

from typing import Callable, Dict, Type

from .base import SandboxRuntime


_RUNTIME_REGISTRY: Dict[str, Type[SandboxRuntime]] = {}


def register_runtime(name: str) -> Callable[[Type[SandboxRuntime]], Type[SandboxRuntime]]:
    """注册一个 runtime provider，避免 evaluator 维护平台分支。"""
    def decorator(runtime_cls: Type[SandboxRuntime]) -> Type[SandboxRuntime]:
        _RUNTIME_REGISTRY[name] = runtime_cls
        return runtime_cls

    return decorator


def create_runtime(name: str) -> SandboxRuntime:
    """按名称创建 runtime。内置 provider 会在模块导入时完成注册。"""
    runtime_cls = _RUNTIME_REGISTRY.get(name)
    if runtime_cls is None:
        raise ValueError(f"unsupported runtime backend: {name}")
    return runtime_cls()


# 导入内置 provider 以触发装饰器注册。
from . import providers as _providers  # noqa: E402,F401
