"""PTDBench 内置 runtime provider。"""

from .docker import DockerRuntime
from .local import LocalRuntime

__all__ = ["DockerRuntime", "LocalRuntime"]
