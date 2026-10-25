"""Unit tests for scripts/cybertron_stage.py (pre-agent staging helper).

Covers the pure, cluster-free surface: CLI parsing, the constructor-kwarg
resolution (attach vs create, image_name / cache_dir folding), the on-devspace
prepare recipe, and the dry-run output. The cluster-touching run_stage() path
is exercised only through its argument plumbing, never by reaching cctl/ssh.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "cybertron_stage.py"
_spec = importlib.util.spec_from_file_location("cybertron_stage", _SCRIPT)
assert _spec and _spec.loader
stage = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stage)


class TestParseKwargs:
    def test_plain_strings_pass_through(self):
        got = stage.parse_kwargs(["project=xinfer-chenyinghao", "cluster=wuxi_train"])
        assert got == {"project": "xinfer-chenyinghao", "cluster": "wuxi_train"}

    def test_known_int_fields_are_coerced(self):
        # cpu/memory_gb reach the constructor as ints, not "16"/"128".
        got = stage.parse_kwargs(["cpu=16", "memory_gb=128", "project=p"])
        assert got["cpu"] == 16 and got["memory_gb"] == 128
        assert isinstance(got["cpu"], int) and isinstance(got["memory_gb"], int)

    def test_value_may_contain_equals(self):
        got = stage.parse_kwargs(["image_map_path=/a/b=c.json"])
        assert got["image_map_path"] == "/a/b=c.json"

    def test_bad_pair_is_rejected(self):
        with pytest.raises(ValueError):
            stage.parse_kwargs(["not-a-pair"])

    def test_priority_passes_through_as_string(self):
        got = stage.parse_kwargs(["priority=PREEMPTABLE"])
        assert got == {"priority": "PREEMPTABLE"}

    def test_requeue_bool_is_coerced(self):
        # Without coercion "false" would be a truthy non-empty string.
        assert stage.parse_kwargs(["requeue=true"])["requeue"] is True
        assert stage.parse_kwargs(["requeue=false"])["requeue"] is False
        assert stage.parse_kwargs(["requeue=1"])["requeue"] is True
        assert stage.parse_kwargs(["requeue=0"])["requeue"] is False

    def test_requeue_rejects_non_boolean(self):
        with pytest.raises(ValueError, match="expects a boolean"):
            stage.parse_kwargs(["requeue=maybe"])


class TestConstructorKwargs:
    def _args(self, **over):
        argv = [over.pop("task", "tasks/x")]
        for pair in over.pop("environment_kwarg", []):
            argv += ["--environment-kwarg", pair]
        for flag, key in (("--image", "image"), ("--cache-dir", "cache_dir")):
            if key in over:
                argv += [flag, str(over.pop(key))]
        return stage.parse_args(argv)

    def test_image_flag_becomes_image_name(self):
        kwargs = stage.constructor_kwargs(self._args(image="slime-v0.3.2"))
        assert kwargs["image_name"] == "slime-v0.3.2"

    def test_cache_dir_flag_folds_in(self):
        kwargs = stage.constructor_kwargs(self._args(cache_dir="/user/me/cache"))
        assert kwargs["cache_dir"] == "/user/me/cache"

    def test_explicit_pair_wins_over_flag(self):
        # An image_name passed as a pair is not clobbered by --image.
        kwargs = stage.constructor_kwargs(
            self._args(image="aaa", environment_kwarg=["image_name=bbb"])
        )
        assert kwargs["image_name"] == "bbb"

    def test_devspace_id_signals_attach_mode(self):
        kwargs = stage.constructor_kwargs(
            self._args(environment_kwarg=["devspace_id=tasks/123"])
        )
        assert kwargs["devspace_id"] == "tasks/123"


class TestStageRecipe:
    def test_installs_tooling_before_running_prepare(self):
        recipe = stage.build_stage_recipe()
        assert "install-ptdbench-tooling.sh" in recipe
        assert recipe.index("install-ptdbench-tooling.sh") < recipe.index(
            "bash prepare.sh"
        )

    def test_exports_cache_and_workspace_env(self):
        recipe = stage.build_stage_recipe()
        assert "PTDBENCH_WORKSPACE=/workspace" in recipe
        assert "PTDBENCH_MODELS=/ptdbench/models" in recipe
        assert "PTDBENCH_DATASETS=/ptdbench/datasets" in recipe
        assert "PTDBENCH_STAGE=prepare" in recipe

    def test_cds_into_the_uploaded_environment(self):
        recipe = stage.build_stage_recipe()
        assert "cd /workspace/.ptdbench-environment" in recipe

    def test_proxy_and_hf_are_included_when_set(self):
        recipe = stage.build_stage_recipe(
            http_proxy="http://proxy:7891", hf_endpoint="https://hf-mirror.com"
        )
        assert "http://proxy:7891" in recipe
        assert "HF_ENDPOINT=https://hf-mirror.com" in recipe

    def test_proxy_and_hf_are_omitted_when_unset(self):
        recipe = stage.build_stage_recipe(http_proxy=None, hf_endpoint=None)
        assert "http_proxy" not in recipe
        assert "HF_ENDPOINT" not in recipe


class TestEnvConfig:
    def test_carries_gpus_and_image(self):
        cfg = stage.build_env_config("verl", 8)
        assert cfg.gpus == 8
        assert cfg.docker_image == "verl"


class TestDryRun:
    def test_dry_run_prints_plan_and_touches_nothing(self, capsys):
        rc = stage.main(["tasks/x", "--gpu", "8", "--image", "slime-v0.3.2-cu129"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "dry-run" in out
        assert "mode:    create" in out

    def test_dry_run_reports_attach_mode(self, capsys):
        rc = stage.main(["tasks/x", "--environment-kwarg", "devspace_id=tasks/123"])
        assert rc == 0
        assert "mode:    attach" in capsys.readouterr().out
