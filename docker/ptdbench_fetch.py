#!/usr/bin/env python3
"""Fetch model weights and datasets from inside the prepare stage.

Installed into the image as `ptdbench_fetch_hf` and `ptdbench_fetch_url`, and
called by each task's `environment/prepare.sh`. This is the in-container
counterpart of `ptdbench/assets/resolver.py` and keeps its two load-bearing
rules:

* **Cache first.** A repo+revision already in `$PTDBENCH_MODELS` (models) or
  `$PTDBENCH_DATASETS` (datasets) is reused, so the second task that wants
  Qwen2.5-0.5B does not re-download a gigabyte, and a rebuilt image does not
  either.
* **A checksum mismatch is a hard failure, never a silent re-download.** Data
  that does not match what the task recorded is either corrupt or altered;
  turning that into a retry line in a log is how a poisoned or truncated weight
  file reaches a training run nobody is watching.

It is deliberately standalone -- no `ptdbench` import. It runs inside the
evaluation image the task declared (docker/images/<name>.Dockerfile), and that
image contains the task's stack, not the harness.

Usage, as prepare.sh writes it:

    ptdbench_fetch_hf  --repo Qwen/Qwen2.5-0.5B-Instruct \\
                       --revision <40-char-sha> \\
                       --dest "$PTDBENCH_PREPARED/models/qwen2.5-0.5b-instruct" \\
                       --expect model.safetensors=<sha256> \\
                       --expect-size model.safetensors=988097824

    ptdbench_fetch_url --url https://example.org/data.tar.gz \\
                       --dest "$PTDBENCH_PREPARED/data/gsm8k" \\
                       --sha256 <sha256> --extract
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence

_CHUNK = 1024 * 1024

# 两个缓存根，宿主机上就是仓库根的 models/ 和 datasets/。默认值必须和
# ptdbench/core/types.py 的 MODELS_DIR / DATASETS_DIR、以及
# docker/ptdbench-activate.sh 里的一致。
MODELS_ROOT = Path(os.environ.get("PTDBENCH_MODELS", "/ptdbench/models"))
DATASETS_ROOT = Path(os.environ.get("PTDBENCH_DATASETS", "/ptdbench/datasets"))


def root_for(repo_type: str) -> Path:
    """按资产种类选缓存根。

    分两个根而不是一个带子目录的，是因为宿主机上这是两个并排的、看得见的目录：
    漏挂一个会立刻表现为"下载没留下来"，而不是安静地写进容器可写层。
    """
    return DATASETS_ROOT if repo_type == "dataset" else MODELS_ROOT


class FetchError(RuntimeError):
    """The asset is missing, unverifiable, or could not be downloaded."""


# --------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    """Block-wise, because weight files run to several GB."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(directory: Path, expect: Dict[str, str], sizes: Dict[str, int]) -> List[str]:
    """Check the declared files under `directory`; return problems, empty if ok.

    Only files the task actually declared are checked. Tokenizer and config
    files are pinned by the revision instead -- hashing every small file in a
    snapshot buys little and makes the task package tedious to write.
    """
    problems: List[str] = []
    for relative in sorted(set(expect) | set(sizes)):
        target = directory / relative
        if not target.exists():
            problems.append(f"{relative}: missing")
            continue
        if not target.is_file():
            problems.append(f"{relative}: expected a file")
            continue
        expected_size = sizes.get(relative)
        if expected_size is not None:
            actual_size = target.stat().st_size
            if actual_size != expected_size:
                # Size first: it is free, and a truncated download -- by far the
                # most common failure -- is far more legible as a size than as a
                # pair of hex strings.
                problems.append(f"{relative}: size {actual_size} != expected {expected_size}")
                continue
        expected_hash = expect.get(relative)
        if expected_hash:
            actual = sha256_file(target)
            if actual.lower() != expected_hash.lower():
                problems.append(
                    f"{relative}: sha256 mismatch\n"
                    f"      expected {expected_hash}\n"
                    f"      actual   {actual}"
                )
    return problems


def require_verified(directory: Path, expect: Dict[str, str], sizes: Dict[str, int], what: str) -> None:
    problems = verify(directory, expect, sizes)
    if problems:
        raise FetchError(f"{what} at {directory} failed verification:\n    " + "\n    ".join(problems))


# --------------------------------------------------------------------------
# materializing cache -> destination
# --------------------------------------------------------------------------


def materialize(source: Path, dest: Path) -> None:
    """Put the cached tree at `dest`, without paying for a second copy if we can.

    Hard links, falling back to a copy across filesystems. The cache roots and
    `$PTDBENCH_PREPARED` are normally two directories on one mount, so the link
    path is the usual one and a 1 GB checkpoint costs inodes rather than
    gigabytes.

    Not symlinks: the link target is a container path (/ptdbench/models/...),
    and on the harbor path `--dest` is uploaded into Harbor's container as a
    plain copy of the workspace, where a symlink would dangle or point at
    whatever that container happens to mount there. `--dest` has to be real
    bytes the agent and verifier stages can read on their own.

    `source` is a HuggingFace snapshot dir whose entries are themselves
    symlinks into the sibling blobs store (relative targets like
    ``../../blobs/<sha>``). We resolve each to its real blob before hardlinking:
    ``os.link`` does not follow the symlink on every filesystem (notably the
    Cybertron ``/user`` mount), and hardlinking the *symlink* leaves a dangling
    entry at `dest` whose relative target does not resolve from the new
    location. That dangle surfaces far downstream -- an empty ``config.json``
    ("Unrecognized model ... should have a model_type key"), an unreadable
    ``tokenizer.json``, a ``tar: No such file`` -- long after fetch has printed
    "ready". Resolving to the real blob keeps `dest` real bytes everywhere.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest)

    def _link_real(src: str, dst: str) -> None:
        os.link(os.path.realpath(src), dst)

    try:
        shutil.copytree(source, dest, copy_function=_link_real, symlinks=False)
    except OSError:
        # Cross-device, or a filesystem without hard links. copytree's default
        # copy_function open()s the source, which always follows symlinks, so
        # the real bytes are copied and this path never dangles.
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(source, dest, symlinks=False)


# --------------------------------------------------------------------------
# huggingface
# --------------------------------------------------------------------------


def hf_cache_dir(repo: str, revision: str, repo_type: str) -> Path:
    prefix = "datasets--" if repo_type == "dataset" else "models--"
    slug = prefix + repo.replace("/", "--")
    return root_for(repo_type) / "hf" / "hub" / slug / "snapshots" / revision


def fetch_hf(args: argparse.Namespace) -> Path:
    expect, sizes = parse_expectations(args)
    repo_type = args.type
    cached = hf_cache_dir(args.repo, args.revision, repo_type)

    if cached.is_dir():
        # Present already: verify and stop. If it does not verify we fail here
        # rather than re-downloading over it -- see the module docstring.
        require_verified(cached, expect, sizes, f"cached {args.repo}@{args.revision[:12]}")
        print(f"[fetch] cache hit  {args.repo}@{args.revision[:12]} -> {cached}")
    else:
        print(f"[fetch] downloading {args.repo}@{args.revision[:12]}")
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise FetchError(
                f"huggingface_hub is required to fetch {args.repo!r}, and the active "
                f"environment does not have it. Add it to the environment's "
                f"docker/images/*.Dockerfile, or pre-populate {cached}."
            ) from exc
        # cache_dir, not HF_HOME: huggingface_hub freezes HF_HOME into
        # constants.HF_HUB_CACHE at import time, so assigning to os.environ here
        # -- after the import above -- is a no-op, and every download silently
        # went to ~/.cache/huggingface in the container's writable layer instead.
        # The argument is the one entry point that still works post-import.
        #
        # The layout is HF_HOME's own ($HF_HOME/hub), so it *is* the cache layout
        # hf_cache_dir() looks in and the hit above needs no bookkeeping file.
        # Per type, so models land under models/ and datasets under datasets/.
        hub_cache = root_for(repo_type) / "hf" / "hub"
        hub_cache.mkdir(parents=True, exist_ok=True)
        downloaded = Path(
            snapshot_download(
                repo_id=args.repo,
                revision=args.revision,
                repo_type=repo_type,
                cache_dir=str(hub_cache),
                max_workers=args.workers,
                # None means "the whole repo". Narrowing matters when a repo
                # carries the same weights in several formats: pulling the .bin
                # alongside the .safetensors doubles the download for bytes
                # nothing reads.
                allow_patterns=args.allow or None,
            )
        )
        require_verified(downloaded, expect, sizes, f"downloaded {args.repo}@{args.revision[:12]}")
        cached = downloaded

    materialize(cached, args.dest)
    return args.dest


# --------------------------------------------------------------------------
# plain URL
# --------------------------------------------------------------------------


def fetch_url(args: argparse.Namespace) -> Path:
    expect, sizes = parse_expectations(args)
    if not args.sha256:
        # A URL is mutable in a way a HF revision is not: same address, different
        # bytes tomorrow. Without a checksum there is nothing that would notice.
        raise FetchError("--sha256 is required for --url: an unpinned URL is not reproducible")

    blob_cache = root_for(args.type) / "url" / args.sha256.lower()
    blob = blob_cache / "blob"
    if blob.is_file():
        actual = sha256_file(blob)
        if actual.lower() != args.sha256.lower():
            raise FetchError(
                f"cached blob for {args.url} is corrupt:\n"
                f"    expected {args.sha256}\n    actual   {actual}\n"
                f"    remove {blob} and re-run"
            )
        print(f"[fetch] cache hit  {args.url} -> {blob}")
    else:
        print(f"[fetch] downloading {args.url}")
        blob_cache.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=blob_cache, delete=False) as handle:
            partial = Path(handle.name)
        try:
            urllib.request.urlretrieve(args.url, partial)  # noqa: S310 - URL comes from the task package
            if args.size is not None and partial.stat().st_size != args.size:
                raise FetchError(
                    f"{args.url}: size {partial.stat().st_size} != expected {args.size}"
                )
            actual = sha256_file(partial)
            if actual.lower() != args.sha256.lower():
                raise FetchError(
                    f"{args.url}: sha256 mismatch\n"
                    f"    expected {args.sha256}\n    actual   {actual}"
                )
            # Rename only after it verifies, so an interrupted download can
            # never be mistaken for a cache hit next time.
            partial.rename(blob)
        finally:
            partial.unlink(missing_ok=True)

    args.dest.parent.mkdir(parents=True, exist_ok=True)
    if args.extract:
        if args.dest.exists():
            shutil.rmtree(args.dest)
        args.dest.mkdir(parents=True)
        extract(blob, args.dest)
        require_verified(args.dest, expect, sizes, f"extracted {args.url}")
    else:
        if args.dest.exists():
            args.dest.unlink()
        try:
            os.link(blob, args.dest)
        except OSError:
            shutil.copyfile(blob, args.dest)
    return args.dest


def extract(blob: Path, target: Path) -> None:
    """Unpack into `target`.

    Member names inside an archive are external input: a member called `../x`
    or `/etc/x` writes outside `target`. tarfile's `filter="data"` exists for
    exactly that (CVE-2007-4559); zip has no equivalent, so it gets a manual
    pass first.
    """
    if tarfile.is_tarfile(blob):
        with tarfile.open(blob) as tar:
            tar.extractall(target, filter="data")
        return
    if zipfile.is_zipfile(blob):
        resolved_target = target.resolve()
        with zipfile.ZipFile(blob) as archive:
            for member in archive.namelist():
                destination = (target / member).resolve()
                if destination != resolved_target and resolved_target not in destination.parents:
                    raise FetchError(f"archive member {member!r} escapes the extraction directory")
            archive.extractall(target)
        return
    raise FetchError(f"unrecognized archive format: {blob}")


# --------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------


def parse_expectations(args: argparse.Namespace) -> tuple[Dict[str, str], Dict[str, int]]:
    expect: Dict[str, str] = {}
    sizes: Dict[str, int] = {}
    for item in args.expect or []:
        relative, _, value = item.partition("=")
        if not relative or not value:
            raise FetchError(f"--expect wants `<relative-path>=<sha256>`, got {item!r}")
        expect[relative] = value
    for item in args.expect_size or []:
        relative, _, value = item.partition("=")
        if not relative or not value.isdigit():
            raise FetchError(f"--expect-size wants `<relative-path>=<bytes>`, got {item!r}")
        sizes[relative] = int(value)
    return expect, sizes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ptdbench_fetch", description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="mode", required=True)

    def add_common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--dest", required=True, type=Path, help="where the task expects it")
        sub.add_argument(
            "--expect",
            action="append",
            metavar="PATH=SHA256",
            help="declared checksum, relative to the fetched tree; repeatable",
        )
        sub.add_argument(
            "--expect-size",
            action="append",
            metavar="PATH=BYTES",
            help="declared size, relative to the fetched tree; repeatable",
        )

    hf = subparsers.add_parser("hf", help="fetch a HuggingFace repo at a pinned revision")
    hf.add_argument("--repo", required=True)
    hf.add_argument(
        "--revision",
        required=True,
        help="a full commit sha; a branch or tag name is not a pin",
    )
    hf.add_argument("--type", choices=("model", "dataset"), default="model")
    hf.add_argument(
        "--allow",
        action="append",
        metavar="GLOB",
        help=(
            "restrict the snapshot to these paths; repeatable. Omit to take the "
            "whole repo"
        ),
    )
    hf.add_argument("--workers", type=int, default=8)
    add_common(hf)

    url = subparsers.add_parser("url", help="fetch a file or archive over http(s)")
    url.add_argument("--url", required=True)
    url.add_argument("--sha256", required=True)
    url.add_argument("--size", type=int, default=None)
    url.add_argument(
        "--type",
        choices=("model", "dataset"),
        default="dataset",
        help=(
            "which cache root holds the blob; defaults to dataset because that "
            "is what plain URLs almost always are"
        ),
    )
    url.add_argument("--extract", action="store_true", help="unpack into --dest")
    add_common(url)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.mode == "hf":
            if len(args.revision) != 40:
                raise FetchError(
                    f"--revision {args.revision!r} is not a 40-char commit sha; a branch "
                    "or tag can move under the benchmark and change its ground truth"
                )
            dest = fetch_hf(args)
        else:
            dest = fetch_url(args)
    except FetchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"[fetch] ready {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
