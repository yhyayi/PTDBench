from __future__ import annotations

from typing import Optional

from ...core.types import CommandResult
from ...runtimes import CommandSpec, EnvironmentArtifact, SandboxRuntime
from ..base import AgentExecutionContext, AgentRunner
from ..factory import register_agent_runner


@register_agent_runner("shell")
class ShellAgentRunner(AgentRunner):
    # 最轻量的兜底 runner：直接执行一条 shell 命令。

    def run(
        self,
        context: AgentExecutionContext,
        *,
        runtime: SandboxRuntime,
        artifact: Optional[EnvironmentArtifact],
    ) -> CommandResult:
        command = self.spec.command
        if not command:
            raise ValueError("shell agent needs command")

        request = context.sandbox_request(
            runtime,
            network=self.spec.network,
        )
        result = runtime.run(request, CommandSpec(shell=command), artifact=artifact)
        self._write_agent_trace(
            context.results,
            {
                "kind": self.kind,
                "backend": runtime.name,
                "command": command,
                "cwd": result.cwd,
                "env": self._redact_env(runtime.standard_env(request)),
                "result": result.to_dict(),
            },
        )
        return result
