#!/usr/bin/env python3
"""Regroup the tasks' requirements into the conda environment registry.

Each task declares `environment.conda: <name>` and ships its own complete
`environment/requirements.txt`. This tool checks that every task claiming the
same environment ships the *same* closure, and writes the agreed result to
`docker/envs/<name>/requirements.txt`, which is what the image build installs.

Equality, not union. Taking the union would mean adding a task can change an
existing task's environment, and for a benchmark that grades by reproducing a
specific training failure that rewrites ground truth without anything failing.
A task that genuinely needs a different closure needs a different environment
name; a task that needs one or two extra pure-Python packages needs
`environment/prepare.sh`.

    python3 docker/collect_envs.py            # rewrite docker/envs/*/requirements.txt
    python3 docker/collect_envs.py --check    # verify only; non-zero if stale

`--check` is what docker/build.sh runs: it is docker-free and takes
milliseconds, so a registry that has drifted fails before the multi-hour build
starts rather than after it.
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ptdbench.utils.yaml import SimpleYamlError, load_simple_yaml  # noqa: E402

ENVS_DIR = REPO_ROOT / "docker" / "envs"
TASKS_DIR = REPO_ROOT / "tasks"
DOCKERFILE = REPO_ROOT / "docker" / "Dockerfile"

# A pinned requirement: an exact version, or a URL that names an immutable
# revision. Anything looser (`>=`, a bare name, a `file://` path left over from
# a build machine) makes the environment unreproducible, which is the one
# property the whole design is buying.
_EXACT = re.compile(r"^[A-Za-z0-9._-]+(\[[A-Za-z0-9,._-]+\])?==[^\s;]+(\s*;.*)?$")
_URL_REQ = re.compile(r"^[A-Za-z0-9._-]+(\[[A-Za-z0-9,._-]+\])?\s*@\s*(?P<url>\S+)(\s*;.*)?$")
_GIT_PINNED = re.compile(r"^git\+[^@\s]+@[0-9a-fA-F]{40}(#.*)?$")
_ARCHIVE = re.compile(r"^https?://\S+\.(whl|tar\.gz|zip)(#.*)?$")
_OPTION = re.compile(r"^--(extra-index-url|index-url|find-links)\s+\S+$")


class RegistryError(RuntimeError):
    """The registry cannot be derived from the tasks as they stand."""


def strip_comments(text: str) -> List[str]:
    """Requirement lines only: no comments, no blank lines, original order."""
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


def sort_key(line: str) -> Tuple[int, str]:
    """Options first, then package name, case- and separator-insensitive.

    pip treats `_` and `-` as the same character in a distribution name, so
    `torch_memory_saver` and `torch-memory-saver` must not sort to different
    places -- otherwise two tasks that agree on content would disagree on order.
    """
    if line.startswith("-"):
        return (0, line)
    name = re.split(r"[\s@=<>!~\[;]", line, maxsplit=1)[0]
    return (1, name.lower().replace("_", "-"))


def normalize(text: str) -> List[str]:
    return sorted(strip_comments(text), key=sort_key)


def check_pinned(env: str, source: str, lines: Sequence[str]) -> List[str]:
    problems = []
    for line in lines:
        if _OPTION.match(line):
            continue
        if line.startswith("-e ") or line.startswith("--editable"):
            problems.append(
                f"{line!r}: editable installs belong in docker/envs/{env}/sources.txt, "
                "not in requirements.txt"
            )
            continue
        if _EXACT.match(line):
            continue
        match = _URL_REQ.match(line)
        if match:
            url = match.group("url")
            if _GIT_PINNED.match(url) or _ARCHIVE.match(url):
                continue
            if url.startswith("file://"):
                problems.append(
                    f"{line!r}: a file:// path is a build machine's local directory, "
                    "not something a rebuild can resolve; use the git ref it was built from"
                )
                continue
            problems.append(
                f"{line!r}: a URL requirement must name a full 40-char commit or a "
                "release archive"
            )
            continue
        problems.append(f"{line!r}: not pinned; requirements must use `==` or a pinned URL")
    return [f"docker/envs/{env}/requirements.txt via {source}: {item}" for item in problems]


def task_metadata_files() -> List[Path]:
    if not TASKS_DIR.is_dir():
        return []
    return sorted(
        path
        for path in TASKS_DIR.rglob("metadata.yaml")
        # tasks/raw/ is the pre-migration dump, not a loadable task package.
        if "raw" not in path.relative_to(TASKS_DIR).parts
    )


def collect() -> Dict[str, List[Tuple[str, str]]]:
    """env name -> [(task id, requirements text)], for tasks on the new schema."""
    groups: Dict[str, List[Tuple[str, str]]] = {}
    for metadata_path in task_metadata_files():
        try:
            data = load_simple_yaml(str(metadata_path))
        except SimpleYamlError as exc:
            raise RegistryError(f"{_rel(metadata_path)}: {exc}") from exc
        environment = (data or {}).get("environment")
        if not isinstance(environment, dict):
            continue
        env = environment.get("conda")
        if not env:
            # Legacy task: still on environment.dockerfile. Not our business.
            continue
        env = str(env).strip()
        task_dir = metadata_path.parent
        declared = str(environment.get("requirements") or "environment/requirements.txt")
        requirements_path = task_dir / declared
        if not requirements_path.is_file():
            raise RegistryError(
                f"{_rel(metadata_path)}: declares environment.conda: {env} but "
                f"{declared} does not exist"
            )
        groups.setdefault(env, []).append(
            (_rel(task_dir), requirements_path.read_text(encoding="utf-8"))
        )
    return groups


def agree(env: str, members: Sequence[Tuple[str, str]]) -> List[str]:
    """The single closure the group agrees on, or raise showing who disagrees."""
    reference_task, reference_text = members[0]
    reference = normalize(reference_text)
    for task, text in members[1:]:
        candidate = normalize(text)
        if candidate == reference:
            continue
        diff = "\n".join(
            difflib.unified_diff(
                reference, candidate, fromfile=reference_task, tofile=task, lineterm=""
            )
        )
        raise RegistryError(
            f"environment {env!r}: {task} and {reference_task} ship different "
            f"requirements.\n{diff}\n\n"
            f"Either align them, or give one of the two its own environment name "
            f"(and its own stage in docker/Dockerfile). The union is deliberately "
            f"not taken -- see docker/envs/README.md."
        )
    return reference


def dockerfile_stages(prefix: str = "env-") -> List[str]:
    """Environment names the Dockerfile has a `FROM ... AS <prefix><name>` for."""
    if not DOCKERFILE.is_file():
        return []
    pattern = re.compile(
        rf"^FROM\s+\S+\s+AS\s+{re.escape(prefix)}(\S+)\s*$", re.IGNORECASE | re.MULTILINE
    )
    return pattern.findall(DOCKERFILE.read_text(encoding="utf-8"))


def dockerfile_prefetch_copies(root: str) -> List[str]:
    """Environment names with a `COPY <root>/<name> ...` line in the Dockerfile.

    Each env-* stage copies its own share of the host prefetch directories, and
    a stage that is missing one builds perfectly well -- it just fetches
    everything over the network instead, for hours, on every rebuild. Nothing
    fails, so nothing announces it; hence the check.
    """
    if not DOCKERFILE.is_file():
        return []
    pattern = re.compile(
        rf"^COPY\s+{re.escape(root)}/(\S+)\s+\S+\s*$", re.IGNORECASE | re.MULTILINE
    )
    return pattern.findall(DOCKERFILE.read_text(encoding="utf-8"))


def check_env_dir(env: str, requirements: Sequence[str]) -> List[str]:
    """Verify the hand-maintained half of an environment directory."""
    problems = []
    env_dir = ENVS_DIR / env
    for name in ("conda.yaml", "build-first.txt", "sources.txt"):
        if not (env_dir / name).is_file():
            problems.append(f"docker/envs/{env}/{name} is missing")
    names = {sort_key(line)[1] for line in requirements if not line.startswith("-")}
    # The framework itself must not be installed into its own environment. The
    # upstream images do not ship it -- the task's workspace carries the
    # checkout its recorded outcome came from, and its launch scripts put
    # /workspace first on PYTHONPATH. A framework in site-packages would shadow
    # that with a different version, which rewrites ground truth without
    # anything failing. See docker/envs/README.md.
    if env.split("-")[0] in names or env in names:
        problems.append(
            f"docker/envs/{env}/requirements.txt installs {env!r} itself; the "
            "framework comes from the task's workspace, not the environment"
        )
    build_first = env_dir / "build-first.txt"
    if build_first.is_file():
        for name in strip_comments(build_first.read_text(encoding="utf-8")):
            if name.lower().replace("_", "-") not in names:
                problems.append(
                    f"docker/envs/{env}/build-first.txt names {name!r}, which is not in "
                    "requirements.txt; the pin is read from there"
                )
    build_flags = env_dir / "build-flags.txt"
    if build_flags.is_file():
        for line in strip_comments(build_flags.read_text(encoding="utf-8")):
            name = line.split()[0]
            if name.lower().replace("_", "-") not in names:
                problems.append(
                    f"docker/envs/{env}/build-flags.txt names {name!r}, which is not in "
                    "requirements.txt; the pin is read from there"
                )
    sources = env_dir / "sources.txt"
    source_dirs = set()
    if sources.is_file():
        for line in strip_comments(sources.read_text(encoding="utf-8")):
            parts = line.split()
            if len(parts) != 3 or not re.fullmatch(r"[0-9a-fA-F]{40}", parts[2]):
                problems.append(
                    f"docker/envs/{env}/sources.txt: expected `<dir> <git-url> <40-char commit>`, "
                    f"got {line!r}"
                )
            else:
                source_dirs.add(parts[0])
    # build-env.sh looks for patches under the source tree's *dir name*, so a
    # typo or a renamed tree turns the whole directory into a silent no-op --
    # the build succeeds and ships an unpatched tree. Same reason an empty
    # patches/<dir> is an error there.
    patches = env_dir / "patches"
    if patches.is_dir():
        for sub in sorted(p for p in patches.iterdir() if p.is_dir()):
            if sub.name not in source_dirs:
                problems.append(
                    f"docker/envs/{env}/patches/{sub.name}/ does not match any tree in "
                    f"sources.txt, so build-env.sh would never apply it"
                )
            elif not list(sub.glob("*.patch")):
                problems.append(
                    f"docker/envs/{env}/patches/{sub.name}/ has no *.patch files; "
                    "remove the directory rather than leaving it empty"
                )
    return problems


def render(env: str, requirements: Sequence[str]) -> str:
    """The generated file, header included, byte-for-byte deterministic."""
    index_options = [line for line in requirements if line.startswith("-")]
    body = [line for line in requirements if not line.startswith("-")]
    header = f"""\
# PTDBench conda env `{env}` -- the complete, pinned package closure.
#
# GENERATED, then committed. Do not hand-edit here: every task that declares
# `environment.conda: {env}` ships a byte-identical copy in its own
# environment/requirements.txt, and docker/collect_envs.py fails the build if
# they disagree. Edit the tasks, then re-run `python3 docker/collect_envs.py`.
#
# Provenance and the fixups applied to the raw `pip freeze` are recorded in
# docker/envs/README.md. Source trees that a task script names by absolute path
# are not here -- they are in sources.txt, installed editable from the env's
# own prefix.
"""
    return header + "\n".join([*index_options, *body]) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the registry matches the tasks; write nothing",
    )
    args = parser.parse_args(argv)

    try:
        groups = collect()
    except RegistryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not groups:
        print("no tasks declare environment.conda yet; nothing to collect")
        return 0

    stages = dockerfile_stages()
    # The two host-prefetch directories each environment gets a COPY of, and the
    # committed placeholder that keeps that COPY resolvable on a fresh clone.
    # See the Dockerfile header.
    prefetch = {
        "wheels": dockerfile_prefetch_copies("wheels"),
        "gitmirror": dockerfile_prefetch_copies("gitmirror"),
    }
    problems: List[str] = []
    stale: List[str] = []

    for env in sorted(groups):
        try:
            requirements = agree(env, groups[env])
        except RegistryError as exc:
            problems.append(str(exc))
            continue
        problems.extend(check_pinned(env, groups[env][0][0], requirements))
        problems.extend(check_env_dir(env, requirements))
        if stages and env not in stages:
            problems.append(
                f"environment {env!r} has no `FROM ... AS env-{env}` stage in "
                "docker/Dockerfile; the build would produce an image without it"
            )
        for root, declared in prefetch.items():
            if declared and env not in declared:
                problems.append(
                    f"environment {env!r} has no `COPY {root}/{env} ...` line in its "
                    f"docker/Dockerfile stage; the build would succeed and fetch "
                    f"everything over the network instead of using docker/{root}/{env}"
                )
            if not (REPO_ROOT / "docker" / root / env / ".gitkeep").is_file():
                problems.append(
                    f"docker/{root}/{env}/.gitkeep is missing; the directory is "
                    f"otherwise gitignored, and `COPY {root}/{env}` on a fresh clone "
                    "fails with \"failed to compute cache key: not found\""
                )
        target = ENVS_DIR / env / "requirements.txt"
        rendered = render(env, requirements)
        current = target.read_text(encoding="utf-8") if target.is_file() else None
        if current == rendered:
            continue
        if args.check:
            stale.append(f"docker/envs/{env}/requirements.txt is out of date")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(rendered, encoding="utf-8")
            print(f"wrote docker/envs/{env}/requirements.txt")

    # A registry directory with no tasks pointing at it is dead weight that the
    # build would still spend an hour on.
    for env_dir in sorted(p for p in ENVS_DIR.glob("*") if p.is_dir()):
        if env_dir.name not in groups:
            problems.append(
                f"docker/envs/{env_dir.name}/ exists but no task declares "
                f"environment.conda: {env_dir.name}"
            )

    for item in problems + stale:
        print(f"error: {item}", file=sys.stderr)
    if stale:
        print(
            "hint: run `python3 docker/collect_envs.py` to regenerate",
            file=sys.stderr,
        )
    if problems or stale:
        return 1

    summary = ", ".join(f"{env} ({len(groups[env])} tasks)" for env in sorted(groups))
    print(f"registry ok: {summary}")
    return 0


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
