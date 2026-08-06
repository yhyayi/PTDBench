from __future__ import annotations

from pathlib import Path
from typing import Optional

from ...executor import CommandSpec, DockerExecutor, DockerMount, LocalExecutor
from ...types import CommandResult
from ..base import AgentExecutionContext, AgentRunner
from ..factory import register_agent_runner


@register_agent_runner("shell")
class ShellAgentRunner(AgentRunner):
    # 最轻量的兜底 runner：直接执行一条 shell 命令。

    def run(
        self,
        context: AgentExecutionContext,
        *,
        local: LocalExecutor,
        docker: DockerExecutor,
        image_tag: Optional[str],
    ) -> CommandResult:
        command = self.spec.command
        if not command:
            raise ValueError("shell agent needs command")

        if self.backend == "local":
            spec = CommandSpec(
                shell=command,
                cwd=context.local_workspace,
                env=context.local_env,
                timeout_s=context.timeout_s,
            )
            result = local.run(spec)
            self._write_agent_trace(
                context.local_results,
                {
                    "kind": self.kind,
                    "backend": self.backend,
                    "command": command,
                    "cwd": result.cwd,
                    "env": self._redact_env(context.local_env),
                    "result": result.to_dict(),
                },
            )
            return result

        if image_tag is None:
            raise ValueError("docker shell agent needs an environment image")

        spec = CommandSpec(
            shell=command,
            cwd=Path(context.container_workspace),
            env=context.container_env,
            timeout_s=context.timeout_s,
        )
        mounts = [
            DockerMount(context.local_workspace, context.container_workspace, read_only=False),
            DockerMount(context.local_visible_case, context.container_visible_case, read_only=True),
            DockerMount(context.local_results, context.container_results, read_only=False),
        ]
        result = docker.run(
            image_tag,
            spec,
            mounts=mounts,
            workdir=context.container_workspace,
            network=self.spec.network,
        )
        self._write_agent_trace(
            context.local_results,
            {
                "kind": self.kind,
                "backend": self.backend,
                "command": command,
                "cwd": result.cwd,
                "env": self._redact_env(context.container_env),
                "result": result.to_dict(),
            },
        )
        return result
