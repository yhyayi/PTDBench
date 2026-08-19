from __future__ import annotations

from typing import Optional

from ...core.types import CommandResult
from ...runtimes import CommandSpec, EnvironmentArtifact, SandboxInstance, SandboxRuntime
from ..base import AgentExecutionContext, AgentRunner
from ..factory import register_agent_runner


@register_agent_runner("shell")
class ShellAgentRunner(AgentRunner):
    # 最轻量的兜底 runner：直接执行一条 shell 命令。

    def setup(
        self,
        context: AgentExecutionContext,
        *,
        runtime: SandboxRuntime,
        sandbox: SandboxInstance,
    ) -> CommandResult:
        return CommandResult([], 0, "", "", False, 0.0, "")

    def run(
        self,
        context: AgentExecutionContext,
        *,
        runtime: SandboxRuntime,
        artifact: Optional[EnvironmentArtifact],
        sandbox: Optional[SandboxInstance] = None,
    ) -> CommandResult:
        command = self.spec.command
        if not command:
            raise ValueError("shell agent needs command")

        request = context.sandbox_request(
            runtime,
            env=dict(self.spec.env),
            network=self.spec.network,
        )
        spec = CommandSpec(shell=command, env=dict(self.spec.env))
        result = (
            runtime.exec(sandbox, spec)
            if sandbox is not None
            else runtime.run(request, spec, artifact=artifact)
        )
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
