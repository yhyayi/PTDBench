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

    async def test_no_match_raises(self, tmp_path, monkeypatch):
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
        with pytest.raises(RuntimeError, match="No Teleport node ending in -1098539"):
            await env._resolve_node_via_tsh()

    async def test_ssh_host_override_short_circuits(self, tmp_path):
        env = _make_env(
            tmp_path, image_id="1", ssh_host="devspace-explicit-9", tsh_path="/opt/tsh"
        )
        assert await env._resolve_devspace_host() == "devspace-explicit-9"


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


class TestSetupDevspaceScript:
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
