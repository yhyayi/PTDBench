"""Unit tests for the Cybertron devspace environment backend."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

import harbor.environments.cybertron as cybertron_mod
from harbor.environments.cybertron import (
    CybertronEnvironment,
    _load_image_map_file,
)
from harbor.environments.factory import _ENVIRONMENT_REGISTRY
from harbor.models.environment_type import EnvironmentType
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths


def _trial_paths(root: Path) -> TrialPaths:
    root.mkdir(parents=True, exist_ok=True)
    trial_paths = TrialPaths(trial_dir=root / "trial")
    trial_paths.mkdir()
    return trial_paths


def _make_env(tmp_path: Path, *, docker_image: str | None = None, **kwargs):
    env_dir = tmp_path / "environment"
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / "Dockerfile").write_text("FROM ubuntu:22.04\n")
    return CybertronEnvironment(
        environment_dir=env_dir,
        environment_name="test-task",
        session_id="test-task__abc123",
        trial_paths=_trial_paths(tmp_path),
        task_env_config=EnvironmentConfig(
            docker_image=docker_image, workdir="/workspace"
        ),
        project="xinfer-chenyinghao",
        **kwargs,
    )


class TestRegistration:
    def test_registered_in_factory(self):
        entry = _ENVIRONMENT_REGISTRY[EnvironmentType.CYBERTRON]
        assert entry.class_name == "CybertronEnvironment"

    def test_type_and_capabilities(self, tmp_path):
        env = _make_env(tmp_path, image_id="25607122")
        assert env.type() is EnvironmentType.CYBERTRON
        caps = env.capabilities
        assert caps.gpus is True
        assert caps.mounted is False


class TestResourceScaling:
    """CPU / memory default to 16 CPU + 152 GB per GPU unless given explicitly."""

    def _env(self, tmp_path, *, gpus, **kwargs):
        env_dir = tmp_path / "environment"
        env_dir.mkdir(parents=True, exist_ok=True)
        (env_dir / "Dockerfile").write_text("FROM ubuntu:22.04\n")
        return CybertronEnvironment(
            environment_dir=env_dir,
            environment_name="test-task",
            session_id="test-task__abc123",
            trial_paths=_trial_paths(tmp_path),
            task_env_config=EnvironmentConfig(
                docker_image=None, workdir="/workspace", gpus=gpus
            ),
            project="xinfer-chenyinghao",
            image_id="1",
            **kwargs,
        )

    def test_scales_with_gpu_count(self, tmp_path):
        env = self._env(tmp_path, gpus=8)
        assert env._cpu == 16 * 8
        assert env._memory_gb == 152 * 8

    def test_zero_gpu_sized_as_one(self, tmp_path):
        env = self._env(tmp_path, gpus=0)
        assert env._cpu == 16
        assert env._memory_gb == 152

    def test_explicit_values_win(self, tmp_path):
        env = self._env(tmp_path, gpus=8, cpu=24, memory_gb=256)
        assert env._cpu == 24
        assert env._memory_gb == 256


class TestPriority:
    """`--priority` (and optional `--requeue`) reach `cctl devspace create`."""

    async def _captured_create_args(self, env, monkeypatch) -> list[str]:
        captured: dict[str, list[str]] = {}

        async def fake_cctl(self, args, *a, **kw):
            captured["args"] = args
            return {"name": "tasks/999"}, 0, ""

        monkeypatch.setattr(CybertronEnvironment, "_cctl", fake_cctl)
        await env._create_devspace()
        return captured["args"]

    async def test_defaults_to_normal(self, tmp_path, monkeypatch):
        env = _make_env(tmp_path, image_id="1")
        args = await self._captured_create_args(env, monkeypatch)
        assert "--priority" in args
        assert args[args.index("--priority") + 1] == "NORMAL"
        assert "--requeue" not in args

    async def test_preemptable_is_passed(self, tmp_path, monkeypatch):
        env = _make_env(tmp_path, image_id="1", priority="PREEMPTABLE")
        args = await self._captured_create_args(env, monkeypatch)
        assert args[args.index("--priority") + 1] == "PREEMPTABLE"

    async def test_lowercase_is_normalized(self, tmp_path):
        env = _make_env(tmp_path, image_id="1", priority="preemptable")
        assert env._priority == "PREEMPTABLE"

    async def test_requeue_appends_flag(self, tmp_path, monkeypatch):
        env = _make_env(tmp_path, image_id="1", priority="PREEMPTABLE", requeue=True)
        args = await self._captured_create_args(env, monkeypatch)
        assert "--requeue" in args

    def test_invalid_priority_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="priority must be one of"):
            _make_env(tmp_path, image_id="1", priority="URGENT")


class TestImageResolution:
    def test_builtin_map(self, tmp_path):
        env = _make_env(tmp_path, docker_image="ptdbench/ptdbench-slime-rc0:latest")
        assert env._resolve_image_id() == "25607122"

    def test_explicit_image_id_wins(self, tmp_path):
        env = _make_env(
            tmp_path,
            docker_image="ptdbench/ptdbench-slime-rc0:latest",
            image_id="99999",
        )
        assert env._resolve_image_id() == "99999"

    def test_inline_image_map_overrides_builtin(self, tmp_path):
        env = _make_env(
            tmp_path,
            docker_image="ptdbench/ptdbench-slime-rc0:latest",
            image_map={"ptdbench/ptdbench-slime-rc0:latest": "77777"},
        )
        assert env._resolve_image_id() == "77777"

    def test_image_map_path_merges_and_kwarg_wins(self, tmp_path):
        map_file = tmp_path / "images.json"
        map_file.write_text(json.dumps({"ptdbench/custom:latest": "12345"}))
        env = _make_env(
            tmp_path,
            docker_image="ptdbench/custom:latest",
            image_map_path=map_file,
        )
        assert env._resolve_image_id() == "12345"
        # kwarg dict beats file entry for the same key
        env2 = _make_env(
            tmp_path,
            docker_image="ptdbench/custom:latest",
            image_map_path=map_file,
            image_map={"ptdbench/custom:latest": "54321"},
        )
        assert env2._resolve_image_id() == "54321"

    def test_raw_platform_id_passthrough(self, tmp_path):
        env = _make_env(tmp_path, docker_image="25601234")
        assert env._resolve_image_id() == "25601234"

    def test_unmappable_name_raises(self, tmp_path):
        env = _make_env(tmp_path, image_id="1")  # pass validation
        env._image_id_override = None
        env.task_env_config.docker_image = "mystery-image-no-colon"
        with pytest.raises(ValueError, match="Cannot map image"):
            env._resolve_image_id()

    def test_missing_image_fails_validation(self, tmp_path):
        with pytest.raises(ValueError, match="requires either"):
            _make_env(tmp_path)


class TestImageMapFile:
    def test_loads_and_coerces_to_str(self, tmp_path):
        f = tmp_path / "m.json"
        f.write_text(json.dumps({"a:latest": 111, "b:latest": "222"}))
        assert _load_image_map_file(f) == {"a:latest": "111", "b:latest": "222"}

    def test_non_object_rejected(self, tmp_path):
        f = tmp_path / "m.json"
        f.write_text(json.dumps(["a", "b"]))
        with pytest.raises(ValueError, match="must contain a JSON object"):
            _load_image_map_file(f)


class TestTransportCommand:
    def test_tsh_transport(self, tmp_path):
        env = _make_env(tmp_path, image_id="1", tsh_path="/opt/tsh")
        env._host = "devspace-user-proj-1098539"
        cmd = env._transport_cmd()
        assert cmd == [
            "/opt/tsh",
            "ssh",
            "--cluster=teleport.cybertron.modelbest.co",
            "root@devspace-user-proj-1098539",
        ]

    def test_ssh_fallback_transport(self, tmp_path):
        env = _make_env(tmp_path, image_id="1", use_tsh=False)
        env._host = "devspace-user-proj-1098539"
        cmd = env._transport_cmd()
        assert cmd[0] == "ssh"
        assert cmd[-1] == "devspace-user-proj-1098539"
        assert "BatchMode=yes" in cmd

    def test_transport_requires_resolved_host(self, tmp_path):
        env = _make_env(tmp_path, image_id="1", tsh_path="/opt/tsh")
        with pytest.raises(RuntimeError, match="host is not resolved"):
            env._transport_cmd()


class TestNodeResolutionViaTsh:
    async def test_matches_node_by_id_suffix(self, tmp_path, monkeypatch):
        env = _make_env(
            tmp_path, image_id="1", tsh_path="/opt/tsh", devspace_id="tasks/1098539"
        )
        nodes = [
            {"spec": {"hostname": "devspace-u-p-711108"}},
            {"spec": {"hostname": "devspace-u-p-1098539"}},
        ]
        monkeypatch.setattr(
            cybertron_mod.asyncio, "create_subprocess_exec", AsyncMock()
        )
        monkeypatch.setattr(
            CybertronEnvironment,
            "_collect_process",
            AsyncMock(return_value=(json.dumps(nodes).encode(), b"", 0)),
        )
        assert await env._resolve_node_via_tsh() == "devspace-u-p-1098539"

    async def test_no_match_returns_none_for_retry(self, tmp_path, monkeypatch):
        # cctl reports Running before Teleport registers the node, so a
        # not-yet-present node is transient: resolution returns None and
        # _wait_for_status polls again rather than failing the start.
        env = _make_env(
            tmp_path, image_id="1", tsh_path="/opt/tsh", devspace_id="tasks/1098539"
        )
        monkeypatch.setattr(
            cybertron_mod.asyncio, "create_subprocess_exec", AsyncMock()
        )
        monkeypatch.setattr(
            CybertronEnvironment,
            "_collect_process",
            AsyncMock(return_value=(b"[]", b"", 0)),
        )
        assert await env._resolve_node_via_tsh() is None

    async def test_ssh_host_override_short_circuits(self, tmp_path):
        env = _make_env(
            tmp_path, image_id="1", ssh_host="devspace-explicit-9", tsh_path="/opt/tsh"
        )
        assert await env._resolve_devspace_host() == "devspace-explicit-9"


class TestWaitForStatus:
    """_wait_for_status gates on Running AND the ssh host registering."""

    def _env(self, tmp_path, monkeypatch, **kwargs):
        env = _make_env(tmp_path, image_id="1", devspace_id="tasks/1098539", **kwargs)
        # Never actually sleep between polls.
        monkeypatch.setattr(cybertron_mod.asyncio, "sleep", AsyncMock())
        return env

    async def test_running_and_host_resolved_sets_host(self, tmp_path, monkeypatch):
        env = self._env(tmp_path, monkeypatch)
        monkeypatch.setattr(
            env, "_cctl", AsyncMock(return_value=({"status": "Running"}, 0, ""))
        )
        monkeypatch.setattr(
            env,
            "_resolve_devspace_host",
            AsyncMock(return_value="devspace-u-p-1098539"),
        )
        await env._wait_for_status()
        assert env._host == "devspace-u-p-1098539"

    async def test_waits_for_host_registration_after_running(
        self, tmp_path, monkeypatch
    ):
        # cctl is Running immediately, but the Teleport node is not registered
        # for the first two polls (None) -- _wait_for_status must retry, not fail.
        env = self._env(tmp_path, monkeypatch)
        monkeypatch.setattr(
            env, "_cctl", AsyncMock(return_value=({"status": "Running"}, 0, ""))
        )
        resolve = AsyncMock(side_effect=[None, None, "devspace-u-p-1098539"])
        monkeypatch.setattr(env, "_resolve_devspace_host", resolve)
        await env._wait_for_status()
        assert env._host == "devspace-u-p-1098539"
        assert resolve.await_count == 3

    async def test_terminal_status_raises(self, tmp_path, monkeypatch):
        env = self._env(tmp_path, monkeypatch)
        monkeypatch.setattr(
            env, "_cctl", AsyncMock(return_value=({"status": "Failed"}, 0, ""))
        )
        monkeypatch.setattr(env, "_resolve_devspace_host", AsyncMock())
        with pytest.raises(RuntimeError, match="terminal status Failed"):
            await env._wait_for_status()

    async def test_times_out_when_host_never_registers(self, tmp_path, monkeypatch):
        env = self._env(tmp_path, monkeypatch, status_timeout_sec=0)
        monkeypatch.setattr(
            env, "_cctl", AsyncMock(return_value=({"status": "Running"}, 0, ""))
        )
        monkeypatch.setattr(env, "_resolve_devspace_host", AsyncMock(return_value=None))
        with pytest.raises(RuntimeError, match="did not become ready"):
            await env._wait_for_status()


class TestWrapSandbox:
    def _env(self, tmp_path, **kwargs):
        env = _make_env(tmp_path, image_id="1", **kwargs)
        env._cluster_user = "u"
        return env

    def test_default_tool_dirs_mapped_ro(self, tmp_path):
        wrapped = self._env(tmp_path)._wrap_sandbox("echo hi")
        for tool_dir in ("/sgl-workspace", "/root/Megatron-LM", "/root/slime"):
            assert f"--ro {tool_dir}" in wrapped

    def test_tool_dirs_override(self, tmp_path):
        env = self._env(tmp_path, sandbox_ro_tool_dirs=["/opt/custom"])
        wrapped = env._wrap_sandbox("echo hi")
        assert "--ro /opt/custom" in wrapped
        assert "/sgl-workspace" not in wrapped

    def test_no_proxy_preamble_absent_without_node_ip(self, tmp_path):
        wrapped = self._env(tmp_path)._wrap_sandbox("echo hi")
        assert "no_proxy" not in wrapped

    def test_no_proxy_preamble_includes_node_ip_and_cidr(self, tmp_path):
        env = self._env(tmp_path)
        env._node_ip = "200.20.26.185"
        wrapped = env._wrap_sandbox("echo hi")
        # node IP, derived /16, and localhost forms all present; profile
        # $no_proxy preserved so external hosts still use the proxy.
        assert "200.20.26.185" in wrapped
        assert "200.20.0.0/16" in wrapped
        assert "127.0.0.1" in wrapped
        assert "${no_proxy:+,$no_proxy}" in wrapped
        assert "export no_proxy=" in wrapped
        assert "export NO_PROXY=" in wrapped

    @pytest.mark.parametrize(
        ("env", "expected"),
        [
            # Profile sets only lowercase: uppercase is mirrored, so dotenv
            # loaders cannot fill it with a .env's literal "$https_proxy".
            ({"https_proxy": "http://p:1", "http_proxy": "http://p:1"}, "http://p:1"),
            # An explicit uppercase value is kept.
            ({"https_proxy": "http://p:1", "HTTPS_PROXY": "http://q:2"}, "http://q:2"),
            # No proxy configured: nothing is invented.
            ({}, "<unset>"),
        ],
    )
    def test_preamble_mirrors_uppercase_proxy(self, tmp_path, env, expected):
        import subprocess

        e = self._env(tmp_path)
        e._node_ip = "10.1.2.3"
        script = e._proxy_preamble() + 'echo "${HTTPS_PROXY-<unset>}"'
        out = subprocess.run(
            ["bash", "-c", script],
            env={"PATH": "/usr/bin:/bin", **env},
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        assert out == expected


class TestResolveNodeIp:
    async def test_valid_ip_accepted(self, tmp_path, monkeypatch):
        env = _make_env(tmp_path, image_id="1")
        monkeypatch.setattr(
            CybertronEnvironment,
            "_ssh_exec",
            AsyncMock(
                return_value=cybertron_mod.ExecResult(
                    stdout="200.20.26.185\n", stderr=None, return_code=0
                )
            ),
        )
        assert await env._resolve_node_ip() == "200.20.26.185"

    async def test_garbage_output_rejected(self, tmp_path, monkeypatch):
        env = _make_env(tmp_path, image_id="1")
        monkeypatch.setattr(
            CybertronEnvironment,
            "_ssh_exec",
            AsyncMock(
                return_value=cybertron_mod.ExecResult(
                    stdout="not-an-ip\n", stderr=None, return_code=0
                )
            ),
        )
        assert await env._resolve_node_ip() is None


class TestRunDirName:
    """The per-trial dir is the agent's realpath cwd, so it must not carry the
    task name: session_id is ``{task_name}__{rand}__env`` and task names tend to
    spell out the defect."""

    def test_run_dir_does_not_leak_the_task_name(self, tmp_path):
        env = _make_env(tmp_path, image_id="1")
        env._cluster_user = "u"
        assert "test-task" not in env._run_dir
        assert env._run_dir.startswith("/user/u/harbor-runs/")
        assert env._workspace_dir.startswith(env._run_dir)
        assert env._logs_dir.startswith(env._run_dir)

    def test_run_dir_is_stable_per_instance_and_fresh_per_trial(self, tmp_path):
        a = _make_env(tmp_path / "a", image_id="1")
        b = _make_env(tmp_path / "b", image_id="1")
        a._cluster_user = b._cluster_user = "u"
        assert a._run_dir == a._run_dir
        assert a._run_dir != b._run_dir


class TestSshConnectRetry:
    def _env(self, tmp_path, monkeypatch, results):
        env = _make_env(tmp_path, image_id="1", use_tsh=False)
        env._host = "h"
        calls: list[int] = []

        async def fake_spawn(*args, **kwargs):
            return object()

        async def fake_collect(process, *, timeout_sec, stdin=None):
            calls.append(1)
            return results[len(calls) - 1]

        async def no_sleep(_):
            return None

        monkeypatch.setattr(cybertron_mod.asyncio, "create_subprocess_exec", fake_spawn)
        monkeypatch.setattr(
            CybertronEnvironment, "_collect_process", staticmethod(fake_collect)
        )
        monkeypatch.setattr(cybertron_mod.asyncio, "sleep", no_sleep)
        return env, calls

    async def test_retries_until_the_proxy_dial_succeeds(self, tmp_path, monkeypatch):
        dial = b"failed connecting to host h:0: unable to establish proxy stream\n"
        env, calls = self._env(
            tmp_path, monkeypatch, [(b"", dial, 255), (b"", dial, 255), (b"ok", b"", 0)]
        )
        assert await env._ssh_raw("tar cf - .") == (b"ok", b"", 0)
        assert len(calls) == 3

    async def test_does_not_resend_a_command_that_reached_the_node(
        self, tmp_path, monkeypatch
    ):
        env, calls = self._env(tmp_path, monkeypatch, [(b"", b"tar: no such dir", 2)])
        assert await env._ssh_raw("tar cf - .") == (b"", b"tar: no such dir", 2)
        assert len(calls) == 1

    async def test_gives_up_after_the_last_retry(self, tmp_path, monkeypatch):
        dial = b"ssh: connect to host h port 22: Connection timed out"
        env, calls = self._env(tmp_path, monkeypatch, [(b"", dial, 255)] * 4)
        assert (await env._ssh_raw("true"))[2] == 255
        assert len(calls) == 4


class TestPathKind:
    """A slow Teleport round trip must not turn /workspace into a "file"."""

    def _env(self, tmp_path, monkeypatch, return_code):
        env = _make_env(tmp_path, image_id="1")
        seen: dict[str, object] = {}

        async def fake_exec(self, command, *args, timeout_sec=None, **kwargs):
            seen["timeout_sec"] = timeout_sec
            return cybertron_mod.ExecResult(
                stdout="", stderr="timed out", return_code=return_code
            )

        monkeypatch.setattr(CybertronEnvironment, "exec", fake_exec)
        return env, seen

    async def test_yes_and_no_are_answered(self, tmp_path, monkeypatch):
        env, seen = self._env(tmp_path, monkeypatch, 0)
        assert await env.is_dir("/workspace") is True
        assert seen["timeout_sec"] >= 60
        env, _ = self._env(tmp_path, monkeypatch, 1)
        assert await env.is_dir("/workspace") is False

    @pytest.mark.parametrize("code", [124, 255])
    async def test_inconclusive_check_raises(self, tmp_path, monkeypatch, code):
        env, _ = self._env(tmp_path, monkeypatch, code)
        with pytest.raises(RuntimeError, match="/workspace"):
            await env.is_dir("/workspace")


class TestSetupDevspaceScript:
    async def test_clears_previous_trial_leftovers_first(self, tmp_path, monkeypatch):
        # /tests survives on an attached devspace's rootfs and is --rw in the
        # agent sandbox: the previous verifier's oracle.md would be readable.
        env = _make_env(tmp_path, image_id="1")
        env._cluster_user = "u"
        captured: dict[str, str] = {}

        async def fake_exec(self, command, *args, **kwargs):
            captured["setup"] = command
            return cybertron_mod.ExecResult(stdout="", stderr=None, return_code=0)

        monkeypatch.setattr(CybertronEnvironment, "_ssh_exec", fake_exec)
        await env._setup_devspace()
        setup = captured["setup"]
        cleanup = "rm -rf /tests /solution /tmp/claude-* /tmp/codex-home"
        assert cleanup in setup
        assert setup.index(cleanup) < setup.index("ln -sfnT ")

    async def test_installs_rename_shim_via_ld_so_preload(self, tmp_path, monkeypatch):
        env = _make_env(tmp_path, image_id="1")
        env._cluster_user = "u"
        captured: dict[str, str] = {}

        async def fake_exec(self, command, *args, **kwargs):
            captured["setup"] = command
            return cybertron_mod.ExecResult(stdout="", stderr=None, return_code=0)

        monkeypatch.setattr(CybertronEnvironment, "_ssh_exec", fake_exec)
        await env._setup_devspace()
        setup = captured["setup"]
        assert "/usr/local/lib/llrename_shim.so" in setup
        assert "/etc/ld.so.preload" in setup
        assert "gcc -O2 -shared -fPIC" in setup
        # idempotent install (don't double-append the preload line)
        assert "grep -qxF" in setup

    async def test_replaces_baked_real_dirs_before_symlinking(
        self, tmp_path, monkeypatch
    ):
        # The eval images ship /workspace + /ptdbench/{models,datasets} as real
        # dirs; `ln -sfn` would silently nest inside them. Setup must drop any
        # pre-existing real (non-symlink) dir first, then link with -T so a
        # lingering real dir fails loudly instead of nesting again.
        env = _make_env(tmp_path, image_id="1")
        env._cluster_user = "u"
        captured: dict[str, str] = {}

        async def fake_exec(self, command, *args, **kwargs):
            captured["setup"] = command
            return cybertron_mod.ExecResult(stdout="", stderr=None, return_code=0)

        monkeypatch.setattr(CybertronEnvironment, "_ssh_exec", fake_exec)
        await env._setup_devspace()
        setup = captured["setup"]
        # only real dirs are removed (an existing correct symlink is left alone)
        assert '[ -e "$_link" ] && [ ! -L "$_link" ]' in setup
        assert "rm -rf" in setup
        # every mount point is linked with -T (no plain `ln -sfn` that would nest)
        for link in ("/workspace", "/logs", "/ptdbench/models", "/ptdbench/datasets"):
            assert f" {link}\n" in setup
        assert setup.count("ln -sfnT ") == 4
