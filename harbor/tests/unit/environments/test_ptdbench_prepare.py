"""The prepare shell that CybertronEnvironment runs on the devspace."""

from __future__ import annotations

import pytest

from harbor.environments.ptdbench_prepare import (
    REMOTE_PREPARED,
    build_prepare_recipe,
    hf_export,
    proxy_export,
)


@pytest.mark.unit
class TestProxyAndMirrorExports:
    def test_empty_when_unset(self):
        assert proxy_export(None) == ""
        assert proxy_export("") == ""
        assert hf_export(None) == ""
        assert hf_export("") == ""

    def test_proxy_covers_both_cases_and_bypasses_internal_hosts(self):
        exported = proxy_export("http://proxy:7891")
        for name in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
            assert f"{name}=http://proxy:7891" in exported
        # Internal traffic must not be sent through the egress proxy.
        assert ".svc.cluster.local" in exported
        assert ".modelbest.co" in exported

    def test_mirror_disables_xet(self):
        # The mirror does not serve xet, so leaving it on makes fetches fail.
        assert "HF_HUB_DISABLE_XET=1" in hf_export("https://hf-mirror.com")


@pytest.mark.unit
class TestPrepareRecipe:
    def test_installs_tooling_before_running_prepare(self):
        recipe = build_prepare_recipe()
        assert recipe.index("install-ptdbench-tooling.sh") < recipe.index(
            "bash prepare.sh"
        )

    def test_exports_the_layout_prepare_scripts_read(self):
        recipe = build_prepare_recipe()
        assert "PTDBENCH_WORKSPACE=/workspace" in recipe
        assert f"PTDBENCH_PREPARED={REMOTE_PREPARED}" in recipe
        assert "PTDBENCH_MODELS=/ptdbench/models" in recipe
        assert "PTDBENCH_DATASETS=/ptdbench/datasets" in recipe
        assert "PTDBENCH_STAGE=prepare" in recipe

    def test_creates_the_prepared_dir(self):
        assert f"mkdir -p {REMOTE_PREPARED}" in build_prepare_recipe()

    def test_task_env_cannot_override_the_backend_layout(self):
        """A generated task.toml pins the *container* paths for agent/verifier.

        Letting those win here would point prepare at /ptdbench on the ephemeral
        container overlay instead of the persistent workspace, so everything it
        downloaded would vanish with the devspace.
        """
        recipe = build_prepare_recipe(
            prepared="/user/me/harbor-runs/trial/prepared",
            task_env={
                "PTDBENCH_PREPARED": "/ptdbench/prepared",
                "CUDA_VISIBLE_DEVICES": "0,1,2,3",
            },
        )
        assert recipe.index("PTDBENCH_PREPARED=/ptdbench/prepared") < recipe.index(
            "PTDBENCH_PREPARED=/user/me/harbor-runs/trial/prepared"
        )
        # Everything else the task declares still reaches prepare.
        assert "CUDA_VISIBLE_DEVICES=0,1,2,3" in recipe

    def test_task_env_values_are_quoted(self):
        recipe = build_prepare_recipe(
            task_env={"PTDBENCH_PIP_INDEX_URL": "a b;rm -rf /"}
        )
        assert "'a b;rm -rf /'" in recipe

    def test_script_name_is_configurable_and_quoted(self):
        assert "bash 'odd name.sh'" in build_prepare_recipe(script="odd name.sh")

    def test_offline_switches_are_cleared_after_task_env(self):
        """prepare is the only networked stage, so it must not inherit the
        agent/verifier offline switches the generated task.toml carries.

        Left set, huggingface_hub raises OfflineModeIsEnabled for any repo that
        is not already cached, so a cold cache could never warm up.
        """
        recipe = build_prepare_recipe(
            task_env={
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
            }
        )
        unset_at = recipe.index("unset HF_HUB_OFFLINE")
        assert recipe.index("HF_HUB_OFFLINE=1") < unset_at
        assert unset_at < recipe.index("bash prepare.sh")
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
            assert name in recipe[unset_at:unset_at + 80]

    def test_fails_fast(self):
        # Without -e a failing download would be followed by a "successful"
        # prepare, and the agent would start against an incomplete workspace.
        assert build_prepare_recipe().startswith("set -euo pipefail;")
