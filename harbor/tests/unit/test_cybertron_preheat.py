"""Unit tests for scripts/cybertron_preheat.py (cache preheat helper)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "cybertron_preheat.py"
_spec = importlib.util.spec_from_file_location("cybertron_preheat", _SCRIPT)
assert _spec and _spec.loader
preheat = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(preheat)


class TestBuildEntry:
    def test_exports_cache_env_and_runs_prepare(self):
        entry = preheat.build_entry(
            cache_dir="/user/me/ptdbench-cache",
            prepare_glob="tasks/*/environment/prepare.sh",
        )
        assert "PTDBENCH_MODELS=/user/me/ptdbench-cache/models" in entry
        assert "PTDBENCH_DATASETS=/user/me/ptdbench-cache/datasets" in entry
        assert "mkdir -p /user/me/ptdbench-cache/models" in entry
        assert "for p in tasks/*/environment/prepare.sh" in entry
        assert "set -euo pipefail" in entry

    def test_repo_subdir_cd(self):
        entry = preheat.build_entry(
            cache_dir="/user/me/ptdbench-cache",
            prepare_glob="environment/prepare.sh",
            repo_subdir="ptdbench",
        )
        assert "cd ptdbench;" in entry


class TestBuildCctlCommand:
    def test_git_batch_flags(self):
        args = preheat.parse_args(
            [
                "--project",
                "xinfer-chenyinghao",
                "--image",
                "25607122",
                "--git-path",
                "git@github.com:org/ptdbench.git",
                "--git-ref",
                "main",
            ]
        )
        entry = preheat.build_entry(
            cache_dir="/user/chenyinghao/ptdbench-cache",
            prepare_glob=args.prepare_glob,
        )
        cmd = preheat.build_cctl_command(args, entry)
        assert cmd[:3] == ["cctl", "job", "create"]
        assert "--code-type" in cmd and cmd[cmd.index("--code-type") + 1] == "git"
        assert cmd[cmd.index("--git-path") + 1] == "git@github.com:org/ptdbench.git"
        assert cmd[cmd.index("--git-ref") + 1] == "main"
        assert cmd[cmd.index("--entry") + 1] == entry
        assert cmd[cmd.index("--image") + 1] == "25607122"

    def test_profile_prefixed(self):
        args = preheat.parse_args(
            [
                "--project",
                "p",
                "--image",
                "1",
                "--git-path",
                "g",
                "--profile",
                "thunlp",
            ]
        )
        cmd = preheat.build_cctl_command(args, "entry")
        assert cmd[:3] == ["cctl", "--profile", "thunlp"]

    def test_dry_run_is_default(self):
        args = preheat.parse_args(["--project", "p", "--image", "1", "--git-path", "g"])
        assert args.run is False


class TestMainDryRun:
    def test_main_dry_run_prints_and_does_not_submit(self, capsys, monkeypatch):
        def _boom(*a, **k):  # pragma: no cover - must not be called
            raise AssertionError("subprocess.run should not run in dry-run")

        monkeypatch.setattr(preheat.subprocess, "run", _boom)
        rc = preheat.main(["--project", "p", "--image", "25607122", "--git-path", "g"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "dry-run" in out
        assert "cctl job create" in out
