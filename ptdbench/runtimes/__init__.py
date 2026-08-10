"""评测执行平台。"""

from .base import CommandSpec, EnvironmentArtifact, SandboxRequest, SandboxRuntime
from .factory import create_runtime, register_runtime
from .providers import DockerRuntime, LocalRuntime

__all__ = [
    "CommandSpec",
    "EnvironmentArtifact",
    "SandboxRequest",
    "SandboxRuntime",
    "create_runtime",
    "register_runtime",
    "DockerRuntime",
    "LocalRuntime",
]
