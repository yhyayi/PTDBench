from __future__ import annotations

import os
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import Optional

from ...core.types import CommandResult
from ..base import CommandSpec, EnvironmentArtifact, SandboxRequest, SandboxRuntime
from ..factory import register_runtime


@register_runtime("local")
class LocalRuntime(SandboxRuntime):
    """本机运行时，仅用于开发和 smoke test，不提供安全隔离。"""

    name = "local"

    @property
    def needs_environment_image(self) -> bool:
        return False

    def build_environment(
        self,
        environment_dir: Path,
        *,
        dockerfile: str = "Dockerfile",
        tag_prefix: str = "ptdbench",
    ) -> EnvironmentArtifact:
        # local 模式不构建镜像，environment 仍由 case validator 检查。
        return EnvironmentArtifact(backend=self.name)

    def path(self, request: SandboxRequest, name: str) -> str:
        paths = {
            "case_root": request.host_case_root,
            "visible_case": request.host_visible_case,
            "workspace": request.host_workspace,
            "results": request.host_results,
            "oracle": request.host_case_root / "oracle",
            "verifier": request.host_case_root / "verifier",
        }
        try:
            return str(paths[name].resolve())
        except KeyError as exc:
            raise ValueError(f"unknown sandbox path: {name}") from exc

    def run(
        self,
        request: SandboxRequest,
        spec: CommandSpec,
        *,
        artifact: Optional[EnvironmentArtifact] = None,
    ) -> CommandResult:
        spec = self._merged_spec(request, spec)
        spec = replace(spec, cwd=Path(self.path(request, request.workdir)))
        argv = spec.to_argv()
        env = os.environ.copy()
        env.update({str(key): str(value) for key, value in spec.env.items()})
        start = time.monotonic()
        timed_out = False
        process = subprocess.Popen(
            argv,
            cwd=str(spec.cwd or Path.cwd()),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=spec.timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            stdout, stderr = process.communicate()
        return CommandResult(
            argv=argv,
            returncode=process.returncode if process.returncode is not None else 124,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            duration_s=time.monotonic() - start,
            cwd=str(spec.cwd or Path.cwd()),
        )
