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

    def test_exports_workspace_and_prepared_scratch(self):
        # prepare.sh references PTDBENCH_WORKSPACE / PTDBENCH_PREPARED under
        # `set -u`; both must be exported (as throwaway scratch) or it aborts.
        entry = preheat.build_entry(
            cache_dir="/user/me/ptdbench-cache",
            prepare_glob="tasks/*/environment/prepare.sh",
        )
        assert "PTDBENCH_STAGE=prepare" in entry
        assert 'export PTDBENCH_WORKSPACE="$ws/workspace"' in entry
        assert 'PTDBENCH_PREPARED="$ws/prepared"' in entry
        assert "/user/me/ptdbench-cache/.warm-scratch" in entry

    def test_hf_endpoint_default_and_disable(self):
        # Nodes can't reach huggingface.co; the mirror must be exported.
        entry = preheat.build_entry(
            cache_dir="/user/me/ptdbench-cache",
            prepare_glob="tasks/*/environment/prepare.sh",
            hf_endpoint="https://hf-mirror.com",
        )
        assert "export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1" in entry
        # default flag value routes through the mirror
        assert (
            preheat.parse_args(
                ["--project", "p", "--image", "i", "--git-path", "g"]
            ).hf_endpoint
            == "https://hf-mirror.com"
        )
        # explicit opt-out drops the export entirely
        off = preheat.build_entry(
            cache_dir="/user/me/ptdbench-cache",
            prepare_glob="tasks/*/environment/prepare.sh",
            hf_endpoint="",
        )
        assert "HF_ENDPOINT" not in off

    def test_http_proxy_default_and_disable(self):
        # The mirror is only reachable via the cluster whitelist proxy; a BATCH
        # entry doesn't inherit it from Teleport, so it must be exported.
        proxy = "http://whitelist-proxy.cybertron.svc.cluster.local:7891"
        entry = preheat.build_entry(
            cache_dir="/user/me/ptdbench-cache",
            prepare_glob="tasks/*/environment/prepare.sh",
            hf_endpoint="https://hf-mirror.com",
            http_proxy=proxy,
        )
        assert f"export http_proxy={proxy} https_proxy={proxy}" in entry
        assert "no_proxy=localhost,127.0.0.1,.modelbest.co,.svc.cluster.local" in entry
        # proxy is exported before HF_ENDPOINT so mirror DNS/connect uses it
        assert entry.index("http_proxy=") < entry.index("HF_ENDPOINT=")
        # default flag value uses the whitelist proxy
        assert (
            preheat.parse_args(
                ["--project", "p", "--image", "i", "--git-path", "g"]
            ).http_proxy
            == proxy
        )
        # explicit opt-out drops it entirely
        off = preheat.build_entry(
            cache_dir="/user/me/ptdbench-cache",
            prepare_glob="tasks/*/environment/prepare.sh",
            http_proxy="",
        )
        assert "http_proxy=" not in off

    def test_seeds_task_workspace_tree(self):
        # slime/rlve prepare.sh converts against $PTDBENCH_WORKSPACE/{scripts,tools}
        # shipped in the task's workspace/; preheat must seed that tree.
        entry = preheat.build_entry(
            cache_dir="/user/me/ptdbench-cache",
            prepare_glob="tasks/*/environment/prepare.sh",
        )
        assert 'tw="$(dirname "$(dirname "$p")")/workspace"' in entry
        assert '[ -d "$tw" ] && cp -a "$tw/." "$PTDBENCH_WORKSPACE/"' in entry

    def test_per_task_failure_does_not_abort_batch_but_fails_job(self):
        # One task's failure must not abort warming the rest (the download cache
        # is the deliverable and is written first), but the job must still exit
        # non-zero so it is marked Failed -- a masked exit 0 misled the user into
        # thinking a run with broken prepare steps had succeeded.
        entry = preheat.build_entry(
            cache_dir="/user/me/ptdbench-cache",
            prepare_glob="tasks/*/environment/prepare.sh",
        )
        assert 'if (cd "$(dirname "$p")" && bash "$(basename "$p")"); then' in entry
        assert 'failed="$failed $p"' in entry
        # loop keeps going on failure (tally), then the job exits non-zero
        assert '[ -z "$failed" ] || {' in entry
        assert "exit 1; }" in entry

    def test_installs_latest_tooling_from_clone(self):
        # prepare.sh's ptdbench_fetch_hf / ptdbench_dataset shims exec
        # /usr/local/libexec/*.py, which the Dockerfiles no longer bake. The
        # entry must run docker/install-ptdbench-tooling.sh (from the clone) to
        # install the current tooling before the prepare loop runs the fetches,
        # so a fix reaches a running job without an image rebuild.
        entry = preheat.build_entry(
            cache_dir="/user/me/ptdbench-cache",
            prepare_glob="tasks/*/environment/prepare.sh",
        )
        assert "bash docker/install-ptdbench-tooling.sh" in entry
        # guarded on the installer existing (no-op on an older checkout)
        assert "[ -f docker/install-ptdbench-tooling.sh ]" in entry
        # install happens before the prepare loop runs the fetches
        assert entry.index("install-ptdbench-tooling.sh") < entry.index("for p in")

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
