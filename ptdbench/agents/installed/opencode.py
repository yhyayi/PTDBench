from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from ...core.types import CommandResult
from ...runtimes import CommandSpec, EnvironmentArtifact, SandboxRuntime
from ..base import AgentExecutionContext, AgentRunner
from ..factory import register_agent_runner


@register_agent_runner("opencode")
class OpenCodeAgentRunner(AgentRunner):
    """通过 OpenCode CLI 调用 OpenAI-compatible 模型。"""

    # 使用一个固定的 provider 名称，避免要求每个 case 额外提供 OpenCode 配置文件。
    provider_name = "ptdbench"

    def run(
        self,
        context: AgentExecutionContext,
        *,
        runtime: SandboxRuntime,
        artifact: Optional[EnvironmentArtifact],
    ) -> CommandResult:
        if not self.spec.base_url:
            raise ValueError("opencode agent needs --base-url")

        config = self._build_config()
        env = {"OPENCODE_CONFIG_CONTENT": json.dumps(config, ensure_ascii=False)}
        request = context.sandbox_request(
            runtime,
            env=env,
            network=self.spec.network,
        )
        workspace_path = runtime.path(request, "workspace")
        prompt = self._build_prompt(context, workspace_path)
        result = runtime.run(
            request,
            CommandSpec(argv=self._build_argv(prompt, workspace_path)),
            artifact=artifact,
        )
        self._write_opencode_trace(context.results, prompt, config, result)
        return result

    def _build_config(self) -> Dict[str, Any]:
        return {
            "$schema": "https://opencode.ai/config.json",
            "provider": {
                self.provider_name: {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "PTDBench OpenAI Compatible",
                    "options": {
                        "baseURL": self.spec.base_url,
                        "apiKey": self.spec.api_key or "EMPTY",
                    },
                    "models": {
                        self._model_name(): {
                            "name": self._model_name(),
                            "limit": {
                                "context": 32768,
                                "output": 8192,
                            },
                        }
                    },
                }
            },
        }

    def _build_argv(self, prompt: str, workdir: str) -> list[str]:
        argv = [self.spec.command or "opencode", "run", "--pure"]
        argv.extend(["--dir", workdir, "--format", "json", "--model", self._model_selector()])
        argv.extend(list(self.spec.extra_args))
        argv.append(prompt)
        return argv

    def _model_name(self) -> str:
        # SGLang 当前默认返回 /model；保留斜杠后 selector 会自然变成
        # ptdbench//model。若服务使用 --served-model-name，则直接使用新名称。
        return self.spec.model or "/model"

    def _model_selector(self) -> str:
        return f"{self.provider_name}/{self._model_name()}"

    def _build_prompt(self, context: AgentExecutionContext, workspace_path: str) -> str:
        return "\n".join(
            [
                "You are solving a PTDBench task.",
                "Only modify files under the workspace.",
                "Do not touch oracle/ or verifier/.",
                "Task instruction:",
                context.instruction.strip(),
                "",
                f"Workspace absolute path: {workspace_path}",
                "Use that exact absolute path when calling file tools.",
                "Do not guess or construct a different workspace path.",
                "The task instruction above is the complete public task description.",
            ]
        )

    def _write_opencode_trace(
        self,
        results_dir: Path,
        prompt: str,
        config: Dict[str, Any],
        result: CommandResult,
    ) -> None:
        """把 OpenCode JSON 事件流转换成逐行的 role/content 消息。"""
        metadata = {
            "type": "run_metadata",
            "agent": self.kind,
            "backend": self.backend,
            "provider": self.provider_name,
            "model": self._model_name(),
            "config": self._redact_config(config),
        }
        messages = [
            self._message("system", metadata),
            {"role": "user", "content": prompt},
        ]

        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                messages.append(self._message("system", {"type": "raw_stdout", "text": line}))
                continue
            if not isinstance(event, dict):
                messages.append(self._message("system", {"type": "raw_event", "event": event}))
                continue
            messages.extend(self._event_messages(event))

        execution = result.to_dict()
        execution.pop("stdout", None)
        messages.append(self._message("system", {"type": "execution_result", **execution}))
        self._write_agent_trace_jsonl(results_dir, messages)

    def _event_messages(self, event: Dict[str, Any]) -> list[Dict[str, str]]:
        """将一个 OpenCode event 转成一条或多条标准 message。"""
        event_type = event.get("type")
        part = event.get("part")
        if not isinstance(part, dict):
            return [self._message("system", {"type": "raw_event", "event": event})]

        if event_type == "text":
            text = part.get("text", "")
            return [{"role": "assistant", "content": str(text)}]

        if event_type == "tool_use":
            state = part.get("state")
            state = state if isinstance(state, dict) else {}
            call_id = part.get("callID")
            tool_name = part.get("tool")
            call = {
                "type": "tool_call",
                "tool_call_id": call_id,
                "name": tool_name,
                "arguments": state.get("input", {}),
                "timestamp": event.get("timestamp"),
            }
            tool_result = {
                "type": "tool_result",
                "tool_call_id": call_id,
                "name": tool_name,
                "status": state.get("status"),
                "output": state.get("output"),
                "error": state.get("error"),
                "metadata": state.get("metadata"),
                "title": state.get("title"),
                "time": state.get("time"),
            }
            return [
                self._message("assistant", call),
                self._message("tool", tool_result),
            ]

        # step_start/step_finish 中包含 snapshot、token 用量和停止原因。
        # 这些不是自然语言消息，但属于完整轨迹，因此作为 system 事件保留。
        return [self._message("system", {"type": event_type, "event": event})]

    @staticmethod
    def _message(role: str, content: Any) -> Dict[str, str]:
        if isinstance(content, str):
            text = content
        else:
            text = json.dumps(content, ensure_ascii=False)
        return {"role": role, "content": text}

    @staticmethod
    def _redact_config(config: Dict[str, Any]) -> Dict[str, Any]:
        # trace 中保留连接地址和模型名，但不保存 API key。
        copied = json.loads(json.dumps(config))
        provider = copied.get("provider", {}).get("ptdbench", {})
        options = provider.get("options", {})
        if "apiKey" in options:
            options["apiKey"] = "<redacted>"
        return copied
