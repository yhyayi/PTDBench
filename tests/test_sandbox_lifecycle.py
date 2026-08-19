from __future__ import annotations

import tempfile
import unittest
import subprocess
from unittest.mock import patch
from pathlib import Path

from ptdbench.agents.base import AgentExecutionContext, AgentRunner
from ptdbench.agents.installed.codex import CodexAgentRunner
from ptdbench.cli import build_parser
from ptdbench.core.types import AgentSpec, CommandResult, EvaluationConfig
from ptdbench.evaluation.evaluator import TaskEvaluator
from ptdbench.runtimes import (
    CommandSpec,
    EnvironmentArtifact,
    SandboxInstance,
    SandboxRequest,
    SandboxRuntime,
)
from ptdbench.runtimes.providers.docker import DockerRuntime


def result(label: str, returncode: int = 0) -> CommandResult:
    return CommandResult([label], returncode, label, "", False, 0.0, "/workspace")


class RecordingRuntime(SandboxRuntime):
    name = "recording"

    def __init__(self) -> None:
        self.events: list[str] = []

    @property
    def needs_environment_image(self) -> bool:
        return False

    def build_environment(self, environment_dir, *, dockerfile="Dockerfile", tag_prefix="ptdbench"):
        return EnvironmentArtifact(self.name)

    def path(self, request, name):
        return {
            "task_root": "/task",
            "visible_task": "/task",
            "workspace": "/workspace",
            "results": "/results",
            "oracle": "/task/oracle",
            "verifier": "/task/verifier",
        }[name]

    def run(self, request, spec, *, artifact=None):
        raise AssertionError("persistent lifecycle should be used")

    def create_sandbox(self, request, *, artifact):
        self.events.append(f"create:{request.stage}")
        return SandboxInstance(self.name, request.stage, request)

    def exec(self, sandbox, spec):
        self.events.append(f"exec:{sandbox.request.stage}")
        return result("exec")

    def close_sandbox(self, sandbox):
        self.events.append(f"close:{sandbox.request.stage}")


class RecordingAgent(AgentRunner):
    def __init__(self, setup_returncode: int = 0) -> None:
        super().__init__(AgentSpec(kind="recording"))
        self.setup_returncode = setup_returncode

    def setup(self, context, *, runtime, sandbox):
        runtime.events.append("agent:setup")
        return result("setup", self.setup_returncode)

    def run(self, context, *, runtime, artifact, sandbox=None):
        runtime.events.append("agent:run")
        return result("run")


class SandboxLifecycleTests(unittest.TestCase):
    def context(self, root: Path) -> AgentExecutionContext:
        for name in ("workspace", "visible", "results"):
            (root / name).mkdir()
        return AgentExecutionContext(
            task_id="task",
            instruction="do work",
            metadata={},
            timeout_s=10,
            task_root=root,
            workspace=root / "workspace",
            visible_task=root / "visible",
            results=root / "results",
        )

    def test_setup_and_run_share_one_sandbox(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = RecordingRuntime()
            outcome = TaskEvaluator()._run_agent(
                RecordingAgent(),
                self.context(Path(temp_dir)),
                runtime,
                None,
            )
        self.assertEqual(outcome.setup.returncode, 0)
        self.assertIsNotNone(outcome.execution)
        self.assertEqual(
            runtime.events,
            ["create:agent", "agent:setup", "agent:run", "close:agent"],
        )

    def test_setup_failure_skips_run_and_closes_sandbox(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = RecordingRuntime()
            outcome = TaskEvaluator()._run_agent(
                RecordingAgent(setup_returncode=1),
                self.context(Path(temp_dir)),
                runtime,
                None,
            )
        self.assertIsNone(outcome.execution)
        self.assertEqual(
            runtime.events,
            ["create:agent", "agent:setup", "close:agent"],
        )

    def test_sensitive_docker_env_uses_name_only(self):
        argv, process_env = DockerRuntime._docker_env(
            {"OPENAI_API_KEY": "secret", "PTDBENCH_STAGE": "agent"}
        )
        self.assertEqual(
            argv,
            ["-e", "OPENAI_API_KEY", "-e", "PTDBENCH_STAGE=agent"],
        )
        self.assertEqual(process_env["OPENAI_API_KEY"], "secret")
        self.assertNotIn("secret", " ".join(argv))

    def test_network_defaults_allow_agent_and_verifier_access(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            request = SandboxRequest(
                stage="verifier",
                task_id="task",
                host_task_root=root,
                host_visible_task=root,
                host_workspace=root,
                host_results=root,
            )
        args = build_parser().parse_args(["evaluate", "task"])
        self.assertEqual(request.network, "bridge")
        self.assertEqual(EvaluationConfig().verifier_network, "bridge")
        self.assertFalse(hasattr(args, "agent_backend"))
        self.assertEqual(args.agent_network, "bridge")
        self.assertEqual(args.verifier_network, "bridge")
        judge_args = build_parser().parse_args(
            ["evaluate", "task", "--model", "agent-model", "--judge-model", "judge-model"]
        )
        self.assertEqual(judge_args.model, "agent-model")
        self.assertEqual(judge_args.judge_model, "judge-model")

    def test_evaluate_cli_accepts_stage_env_and_debian_mirror(self):
        args = build_parser().parse_args(
            [
                "evaluate",
                "task",
                "--agent-env",
                "npm_config_registry=https://registry.npmmirror.com",
                "--agent-env",
                "EMPTY_VALUE=",
                "--verifier-env",
                "HTTPS_PROXY=http://proxy:7890",
                "--debian-mirror",
                "https://mirrors.tuna.tsinghua.edu.cn/",
            ]
        )

        self.assertEqual(
            args.agent_env,
            [
                "npm_config_registry=https://registry.npmmirror.com",
                "EMPTY_VALUE=",
            ],
        )
        self.assertEqual(args.verifier_env, ["HTTPS_PROXY=http://proxy:7890"])
        self.assertEqual(args.debian_mirror, "https://mirrors.tuna.tsinghua.edu.cn/")

    def test_docker_verifier_receives_judge_configuration(self):
        env = TaskEvaluator("docker")._verifier_judge_env(
            AgentSpec(
                kind="codex",
                model="agent-model",
                judge_model="judge-model",
                base_url="https://api.example.test/v1",
                api_key="secret-key",
            )
        )
        argv, process_env = DockerRuntime._docker_env(env)
        self.assertEqual(env["PTDBENCH_JUDGE_MODEL"], "judge-model")
        self.assertEqual(env["OPENAI_BASE_URL"], "https://api.example.test/v1")
        self.assertIn("-e", argv)
        self.assertNotIn("secret-key", " ".join(argv))
        self.assertEqual(process_env["OPENAI_API_KEY"], "secret-key")

    def test_verifier_env_is_merged_before_judge_configuration(self):
        class VerifierRuntime(RecordingRuntime):
            def create_sandbox(self, request, *, artifact):
                self.request = request
                return super().create_sandbox(request, artifact=artifact)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for name in ("task", "visible", "workspace", "results", "oracle", "verifier"):
                (root / name).mkdir()
            runtime = VerifierRuntime()
            evaluator = TaskEvaluator("local")
            evaluator.runtime = runtime
            evaluator._run_verifier(
                type("Package", (), {"task_id": "task"})(),
                type(
                    "Evaluation",
                    (),
                    {
                        "root": root / "task",
                        "level": "solution",
                        "oracle": root / "oracle",
                        "verifier": root / "verifier",
                    },
                )(),
                type(
                    "Run",
                    (),
                    {
                        "visible_task": root / "visible",
                        "workspace": root / "workspace",
                        "results": root / "results",
                    },
                )(),
                EvaluationConfig(
                    backend="local",
                    verifier_env={"HTTPS_PROXY": "http://proxy:7890"},
                ),
                EnvironmentArtifact("recording"),
                AgentSpec(
                    kind="codex",
                    model="gpt-5.5",
                    base_url="https://api.example.test/v1",
                ),
            )

        self.assertEqual(runtime.request.env["HTTPS_PROXY"], "http://proxy:7890")
        self.assertEqual(runtime.request.env["PTDBENCH_JUDGE_MODEL"], "gpt-5.5")
        self.assertEqual(
            runtime.request.env["OPENAI_BASE_URL"],
            "https://api.example.test/v1",
        )

    def test_failed_stage_summary_points_to_debug_logs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            results = Path(temp_dir)
            TaskEvaluator("local")._append_command_summary(
                results,
                "agent_setup",
                CommandResult(
                    ["false"],
                    2,
                    "setup output\n",
                    "installation failed\n",
                    False,
                    1.25,
                    "/workspace",
                ),
            )
            run_log = (results / "run.log").read_text(encoding="utf-8")

        self.assertIn("[FAIL] agent_setup exit=2", run_log)
        self.assertIn("error_tail: installation failed", run_log)
        self.assertIn(
            "logs: agent_setup.stdout.log, agent_setup.stderr.log",
            run_log,
        )

    def test_docker_proxy_is_opt_in_for_all_stages(self):
        request = SandboxRequest(
            stage="agent",
            task_id="task",
            host_task_root=Path("/task"),
            host_visible_task=Path("/visible"),
            host_workspace=Path("/workspace"),
            host_results=Path("/results"),
        )
        with patch.dict(
            "os.environ",
            {"PTDBENCH_DOCKER_PROXY": "http://host.docker.internal:7890"},
            clear=False,
        ):
            env = DockerRuntime._proxy_env(request, {"PTDBENCH_STAGE": "agent"})
        self.assertEqual(env["HTTP_PROXY"], "http://host.docker.internal:7890")
        self.assertEqual(env["HTTPS_PROXY"], "http://host.docker.internal:7890")
        self.assertEqual(env["ALL_PROXY"], "http://host.docker.internal:7890")

        verifier_request = SandboxRequest(
            stage="verifier",
            task_id="task",
            host_task_root=Path("/task"),
            host_visible_task=Path("/visible"),
            host_workspace=Path("/workspace"),
            host_results=Path("/results"),
        )
        with patch.dict(
            "os.environ",
            {"PTDBENCH_DOCKER_PROXY": "http://host.docker.internal:7890"},
            clear=False,
        ):
            verifier_env = DockerRuntime._proxy_env(
                verifier_request, {"PTDBENCH_STAGE": "verifier"}
            )
        self.assertEqual(
            verifier_env["HTTPS_PROXY"], "http://host.docker.internal:7890"
        )

    def test_codex_setup_configures_provider_and_logs_in_via_stdin(self):
        class SetupRuntime(RecordingRuntime):
            name = "docker"

            def exec(self, sandbox, spec):
                self.setup_spec = spec
                return result("setup")

        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = SetupRuntime()
            context = self.context(Path(temp_dir))
            runner = CodexAgentRunner(
                AgentSpec(
                    kind="codex",
                    model="gpt-5.5",
                    reasoning_effort="medium",
                    base_url="https://yeysai.com/v1",
                    api_key="secret-key",
                )
            )
            runner.setup(
                context,
                runtime=runtime,
                sandbox=SandboxInstance("recording", "agent", context.sandbox_request(runtime)),
            )

        shell = runtime.setup_spec.shell
        self.assertIn('model_provider = "ptdbench"', shell)
        self.assertIn('model_reasoning_effort = "medium"', shell)
        self.assertIn('base_url = "https://yeysai.com/v1"', shell)
        self.assertIn("codex login --with-api-key", shell)
        self.assertIn('"$OPENAI_API_KEY" | /usr/local/bin/codex login', shell)
        self.assertNotIn("secret-key", shell)
        self.assertEqual(runtime.setup_spec.env["OPENAI_API_KEY"], "secret-key")
        self.assertEqual(runtime.setup_spec.live_log_name, "agent_setup")
        self.assertFalse(runtime.setup_spec.live_log_to_stderr)
        self.assertIn("[setup] downloading nvm", shell)
        self.assertIn("PTDBENCH_DEBIAN_MIRROR", shell)
        self.assertIn("PTDBENCH_NVM_INSTALL_URL", shell)
        self.assertIn("[setup] installing @openai/codex@latest", shell)
        self.assertIn("[setup] complete", shell)

    def test_codex_setup_merges_agent_env_and_debian_mirror(self):
        class SetupRuntime(RecordingRuntime):
            name = "docker"

            def exec(self, sandbox, spec):
                self.setup_spec = spec
                return result("setup")

        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = SetupRuntime()
            context = self.context(Path(temp_dir))
            context = AgentExecutionContext(
                **{
                    **context.__dict__,
                    "debian_mirror": "https://mirrors.tuna.tsinghua.edu.cn",
                }
            )
            runner = CodexAgentRunner(
                AgentSpec(
                    kind="codex",
                    api_key="secret",
                    env={
                        "npm_config_registry": "https://registry.npmmirror.com",
                        "NVM_NODEJS_ORG_MIRROR": "https://npmmirror.com/mirrors/node",
                    },
                )
            )
            runner.setup(
                context,
                runtime=runtime,
                sandbox=SandboxInstance("recording", "agent", context.sandbox_request(runtime)),
            )

        self.assertEqual(
            runtime.setup_spec.env["npm_config_registry"],
            "https://registry.npmmirror.com",
        )
        self.assertEqual(
            runtime.setup_spec.env["NVM_NODEJS_ORG_MIRROR"],
            "https://npmmirror.com/mirrors/node",
        )
        self.assertEqual(
            runtime.setup_spec.env["PTDBENCH_DEBIAN_MIRROR"],
            "https://mirrors.tuna.tsinghua.edu.cn",
        )
        self.assertEqual(runtime.setup_spec.env["OPENAI_API_KEY"], "secret")

    def test_docker_stream_process_writes_live_stage_logs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            process = subprocess.Popen(
                ["sh", "-lc", "echo setup-out; echo setup-error >&2"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            stdout, stderr, timed_out = DockerRuntime._stream_process(
                process,
                timeout_s=5,
                results_dir=Path(temp_dir),
                log_name="agent_setup",
            )
            self.assertFalse(timed_out)
            self.assertEqual(stdout, "setup-out\n")
            self.assertIn("setup-error\n", stderr)
            self.assertEqual(
                (Path(temp_dir) / "agent_setup.stdout.log").read_text(),
                stdout,
            )
            self.assertEqual(
                (Path(temp_dir) / "agent_setup.stderr.log").read_text(),
                stderr,
            )

    def test_codex_setup_debug_enables_terminal_streaming(self):
        class SetupRuntime(RecordingRuntime):
            name = "docker"

            def exec(self, sandbox, spec):
                self.setup_spec = spec
                return result("setup")

        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = SetupRuntime()
            context = self.context(Path(temp_dir))
            context = AgentExecutionContext(
                **{**context.__dict__, "log_level": "debug"}
            )
            runner = CodexAgentRunner(AgentSpec(kind="codex", api_key="secret"))
            runner.setup(
                context,
                runtime=runtime,
                sandbox=SandboxInstance("recording", "agent", context.sandbox_request(runtime)),
            )

        self.assertTrue(runtime.setup_spec.live_log_to_stderr)

    def test_run_directory_prefix_is_readable_and_safe(self):
        prefix = TaskEvaluator._run_directory_prefix(
            "swe_slugfix/001: trial",
            "issue",
        )
        self.assertRegex(
            prefix,
            r"^ptdbench-run-swe_slugfix-001-trial-issue-\d{8}-\d{6}-\d{6}-$",
        )

    def test_debian_mirror_layer_is_inserted_after_from(self):
        dockerfile = "ARG BASE=python:3.11-slim\nFROM ${BASE}\nRUN apt-get update\n"
        materialized = TaskEvaluator._insert_debian_mirror_layer(
            dockerfile,
            "https://mirrors.tuna.tsinghua.edu.cn",
        )

        self.assertLess(
            materialized.index("mirror=https://mirrors.tuna.tsinghua.edu.cn"),
            materialized.index("RUN apt-get update"),
        )
        self.assertIn("deb.debian.org/debian", materialized)

    def test_codex_uses_outer_docker_sandbox(self):
        runner = CodexAgentRunner(
            AgentSpec(kind="codex", reasoning_effort="high")
        )
        argv = runner._build_argv(
            "prompt",
            last_message_path="/results/last-message",
            externally_sandboxed=True,
        )
        self.assertIn("--dangerously-bypass-approvals-and-sandbox", argv)
        self.assertNotIn("--sandbox", argv)
        self.assertIn('model_reasoning_effort="high"', argv)

    def test_codex_prompt_allows_instruction_defined_result_deliverables(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context = self.context(Path(temp_dir))
            prompt = CodexAgentRunner(AgentSpec(kind="codex"))._build_prompt(context)
        self.assertIn("deliverables", prompt)
        self.assertIn("$PTDBENCH_RESULTS", prompt)

    def test_codex_keeps_workspace_sandbox_for_local_runtime(self):
        runner = CodexAgentRunner(AgentSpec(kind="codex"))
        argv = runner._build_argv(
            "prompt",
            last_message_path="/results/last-message",
            externally_sandboxed=False,
        )
        self.assertIn("--sandbox", argv)
        self.assertIn("workspace-write", argv)
        self.assertNotIn("--dangerously-bypass-approvals-and-sandbox", argv)


if __name__ == "__main__":
    unittest.main()
