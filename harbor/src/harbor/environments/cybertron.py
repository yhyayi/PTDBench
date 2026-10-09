"""Cybertron devspace environment backend.

Runs Harbor trials on a long-lived Cybertron *devspace* used as a remote host.

Design (see ``cybertron-harbor-impl-plan.md`` in the ptdbench repo):

* Lifecycle (create / status / stop) goes through the ``cctl`` CLI.
* ``exec`` / ``upload`` / ``download`` go through the devspace's Teleport SSH
  channel (``ssh devspace-<id> "cmd"`` + tar-over-ssh), so no k8s RBAC, no
  ``pods/exec`` and no baked-in exec-agent are required.
* File isolation for the untrusted agent/verifier is done with the Landlock
  LSM via a tiny ``llrun`` launcher (compiled during ``start``). ``mount`` is
  blocked on devspaces (AppArmor), so bubblewrap/apptainer are not usable;
  Landlock needs no mount, no namespace and no capability.

Unlike :class:`~harbor.environments.ec2.EC2Environment` (which runs Docker
Compose *inside* the remote VM), the devspace pod already runs the task image,
so ``exec`` runs commands directly on the devspace over SSH. Only the SSH and
tar-over-ssh primitives are shared in spirit with EC2; the compose machinery is
not used here (``mounted=False``, no docker-compose).
"""

from __future__ import annotations

import asyncio
import json
import shlex
import shutil
import threading
from pathlib import Path, PurePosixPath
from typing import Any, override

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import (
    EnvironmentCapabilities,
    EnvironmentResourceCapabilities,
)
from harbor.environments.tar_transfer import extract_dir_from_bytes, pack_dir_to_bytes
from harbor.models.environment_type import EnvironmentType
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths

# ---------------------------------------------------------------------------
# Defaults for the speech-wx (H100 / wuxi_train) target pool. Every one of
# these can be overridden per-task via ``[environment].kwargs`` or per-job.
# ---------------------------------------------------------------------------
_DEFAULT_CLUSTER = "wuxi_train"
_DEFAULT_RESOURCE_POOL = "speech-wx"
_DEFAULT_BILLING_ACCOUNT_ID = "N00069"
_DEFAULT_GPU_MODEL = "h100"
# Scheduling tier passed to ``cctl devspace create --priority``. NORMAL is a
# quota-backed allocation; PREEMPTABLE borrows idle capacity and can be killed at
# any time (useful on a cluster/pool where you have no NORMAL quota, e.g. running
# on huhehaote without a normal allocation); HIGH is above NORMAL. Default NORMAL
# matches cctl's own default. ``requeue=True`` adds ``--requeue`` so a preempted
# devspace is re-queued rather than lost.
_DEFAULT_PRIORITY = "NORMAL"
_VALID_PRIORITIES = ("HIGH", "NORMAL", "PREEMPTABLE")
# CPU / memory scale with the GPU count by default: an 8-GPU verl FSDP job asks
# Ray for a STRICT_PACK placement group of {GPU:1, CPU:max_colocate_count}*N on a
# single node (max_colocate_count=3 for FSDP), so a flat 16-CPU devspace could
# not satisfy 8*3=24 CPU and the placement group pended forever. Sizing per GPU
# keeps the node proportioned to the job. A GPU-less task still gets one GPU's
# worth (max(gpus, 1)) so it is never provisioned at zero CPU/memory.
_CPU_PER_GPU = 16
_MEMORY_GB_PER_GPU = 152

_DEFAULT_CLUSTER_USER = "chenyinghao"

# Teleport proxy/cluster that fronts the devspaces. Used both as `tsh ssh
# --cluster=` and, in the ssh-config fallback, embedded in the ProxyCommand.
_DEFAULT_TELEPORT_CLUSTER = "teleport.cybertron.modelbest.co"

# Best-effort location of the `tsh` binary shipped by Cursor's Cybertron
# plugin, used only when `tsh` is not already on PATH.
_CURSOR_TSH_FALLBACK = (
    Path.home()
    / "Library"
    / "Application Support"
    / "Cursor"
    / "User"
    / "globalStorage"
    / "yangsuiyun.cybertron"
    / "bin"
    / "tsh"
)

_DEFAULT_STATUS_TIMEOUT_SEC = 900
_DEFAULT_SSH_CONNECT_TIMEOUT_SEC = 600
_DEFAULT_HEARTBEAT_INTERVAL_SEC = 120

_RUNNING_STATUSES = {"Running", "Started", "Ready"}
_TERMINAL_STATUSES = {"Failed", "Killed", "Stopped", "Error", "Cancelled"}

# Built-in ptdbench image tag (``ptdbench/<name>:<tag>``) -> Cybertron platform
# image id map. Extend it per job/task via ``[environment].kwargs.image_map``
# (inline dict) or ``image_map_path`` (a JSON file of the same shape) rather
# than editing this default; precedence is documented on ``_resolve_image_id``.
#
# slime-v0.3.2 was rebuilt on CUDA 12.9 (image_name ptdbench-slime-v0.3.2-cu129,
# id 25607131) to clear the cu130 driver mismatch on wuxi_train/speech-wx (see
# cybertron-cuda13-driver-mismatch). slime-rc0 (cu126, 25607122) and
# verl-vllm017 (cu129, 25607123) run on speech-wx as-is.
_DEFAULT_IMAGE_MAP: dict[str, str] = {
    "ptdbench/ptdbench-slime-rc0:latest": "25607122",
    "ptdbench/ptdbench-slime-v0.3.2-cu129:latest": "25607131",
    "ptdbench/ptdbench-slime-v0.3.2:latest": "25607131",
    "ptdbench/ptdbench-slime:latest": "25607131",
    "ptdbench/ptdbench-verl-vllm017:latest": "25607123",
    "ptdbench/ptdbench-verl:latest": "25607123",
}


def _load_image_map_file(path: str | Path) -> dict[str, str]:
    """Load an image-name -> platform-id map from a JSON file.

    The file must be a JSON object of ``{"<image tag>": "<platform id>"}``;
    values are coerced to ``str`` (ids are numeric on the platform but are
    always passed to ``cctl`` as strings).
    """
    resolved = Path(path).expanduser()
    raw = json.loads(resolved.read_text())
    if not isinstance(raw, dict):
        raise ValueError(
            f"Cybertron image_map_path {resolved} must contain a JSON object "
            f"of {{image: id}}, got {type(raw).__name__}."
        )
    return {str(key): str(value) for key, value in raw.items()}


# Landlock launcher. Confines a process (and all children) to a fixed set of
# paths using raw syscalls (ABI v1, kernel >= 5.13). No mount / namespace /
# capability needed. Built once during start(); see plan §5.3.
_LLRUN_C = r"""
#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <errno.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/prctl.h>
#include <linux/types.h>

#define NR_create   444
#define NR_add      445
#define NR_restrict 446
#define RULE_PATH_BENEATH 1

struct ruleset_attr { __u64 handled_access_fs; };
struct pb_attr { __u64 allowed_access; __s32 parent_fd; } __attribute__((packed));

/* ABI v1: all 13 fs access bits */
#define FS_ALL   0x1fffULL
#define FS_EXEC  (1ULL<<0)
#define FS_WRITE (1ULL<<1)
#define FS_READ  (1ULL<<2)
#define FS_RDIR  (1ULL<<3)

static long l_create(const void*a,size_t s,__u32 f){return syscall(NR_create,a,s,f);}
static long l_add(int fd,const struct pb_attr*a){return syscall(NR_add,fd,RULE_PATH_BENEATH,a,0);}
static long l_restrict(int fd){return syscall(NR_restrict,fd,0);}

static int allow(int rs, const char* path, __u64 acc){
    int fd = open(path, O_PATH|O_CLOEXEC);
    if (fd < 0) return 0;                       /* path missing -> skip */
    struct pb_attr a = { .allowed_access = acc, .parent_fd = fd };
    long r = l_add(rs, &a);
    close(fd);
    if (r) fprintf(stderr, "llrun: add_rule %s: %s\n", path, strerror(errno));
    return 0;
}

int main(int argc, char** argv){
    struct ruleset_attr ra = { .handled_access_fs = FS_ALL };
    int rs = l_create(&ra, sizeof(ra), 0);
    if (rs < 0){ fprintf(stderr,"llrun: create_ruleset: %s\n", strerror(errno)); return 90; }

    const __u64 RO = FS_READ|FS_RDIR|FS_EXEC;
    const char* ro[] = {"/usr","/lib","/lib64","/bin","/sbin","/etc","/opt",
                        "/sys","/run", NULL};
    for (int i=0; ro[i]; i++) allow(rs, ro[i], RO);
    /* /proc must be RW: the NVIDIA driver writes under /proc during CUDA
       context init; RO there yields cudaErrorOperatingSystem (304). */
    const char* rw[] = {"/dev","/tmp","/dev/shm","/var/tmp","/proc", NULL};
    for (int i=0; rw[i]; i++) allow(rs, rw[i], FS_ALL);

    /* parse --ro/--rw ... -- cmd */
    int i = 1;
    for (; i < argc; i++){
        if (!strcmp(argv[i],"--")) { i++; break; }
        else if (!strcmp(argv[i],"--ro") && i+1<argc) allow(rs, argv[++i], RO);
        else if (!strcmp(argv[i],"--rw") && i+1<argc) allow(rs, argv[++i], FS_ALL);
        else { fprintf(stderr,"llrun: bad arg %s\n", argv[i]); return 91; }
    }
    if (i >= argc){ fprintf(stderr,"llrun: no command\n"); return 92; }

    if (prctl(PR_SET_NO_NEW_PRIVS,1,0,0,0)){ perror("prctl"); return 93; }
    if (l_restrict(rs)){ fprintf(stderr,"llrun: restrict_self: %s\n", strerror(errno)); return 94; }

    execvp(argv[i], &argv[i]);
    fprintf(stderr,"llrun: exec %s: %s\n", argv[i], strerror(errno));
    return 95;
}
"""

_LLRUN_BIN = "/usr/local/bin/llrun"
_LLRUN_SRC = "/tmp/llrun.c"

# ---------------------------------------------------------------------------
# Landlock ABI-v1 rename shim (LD_PRELOAD library, installed system-wide via
# /etc/ld.so.preload during setup). Kernel 5.15 has Landlock ABI v1, which
# lacks LANDLOCK_ACCESS_FS_REFER: under a ruleset, a rename()/link() that
# reparents a file to a DIFFERENT directory is unconditionally denied with
# EXDEV -- even when both dirs are allowed. Triton's kernel-cache publish, torch
# inductor, pip, etc. all do os.replace(tmpdir/f, parentdir/f), so GPU workloads
# crash on cross-dir rename during kernel compilation. The shim intercepts
# rename/link and, only on EXDEV, does an atomic copy-into-same-dir-temp +
# same-dir-rename (which ABI v1 permits). Isolation is preserved: the copy's
# open() is still Landlock-checked, so a copy into a denied dir still fails.
#
# It MUST be installed via /etc/ld.so.preload rather than the LD_PRELOAD env
# var: slime's actor_group.py overwrites env_vars["LD_PRELOAD"] with
# torch_memory_saver's .so on the train actor under --offload-train, which would
# evict an LD_PRELOAD-only shim from exactly the process that torch-compiles.
# /etc/ld.so.preload applies to every process IN ADDITION to its own LD_PRELOAD.
_LLRENAME_SHIM_C = r"""
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

typedef int (*rename_fn)(const char *, const char *);
typedef int (*renameat_fn)(int, const char *, int, const char *);
typedef int (*renameat2_fn)(int, const char *, int, const char *, unsigned int);
typedef int (*link_fn)(const char *, const char *);
typedef int (*linkat_fn)(int, const char *, int, const char *, int);

static renameat_fn real_renameat(void) {
    static renameat_fn fn = NULL;
    if (!fn) fn = (renameat_fn)dlsym(RTLD_NEXT, "renameat");
    return fn;
}

/* Copy regular file olddfd/old -> newdfd/new atomically and race-free: copy
   into a temp file IN THE SAME DIRECTORY as `new`, then publish with a
   same-directory rename (which Landlock ABI v1 permits). Readers see either the
   old file or the fully-written new one, never a truncated one. returns 0/-1. */
static int copy_at(int olddfd, const char *old, int newdfd, const char *new) {
    int in = openat(olddfd, old, O_RDONLY | O_CLOEXEC);
    if (in < 0) return -1;
    struct stat st;
    if (fstat(in, &st) != 0) { close(in); return -1; }

    const char *slash = strrchr(new, '/');
    char tmp[PATH_MAX];
    static unsigned long ctr = 0;
    unsigned long seq = __sync_fetch_and_add(&ctr, 1);
    int out = -1;
    for (int attempt = 0; attempt < 64 && out < 0; attempt++) {
        if (slash) {
            int dirlen = (int)(slash - new + 1);
            snprintf(tmp, sizeof tmp, "%.*s.llrename_tmp_%d_%lu", dirlen, new,
                     getpid(), seq + (unsigned long)attempt);
        } else {
            snprintf(tmp, sizeof tmp, ".llrename_tmp_%d_%lu", getpid(),
                     seq + (unsigned long)attempt);
        }
        out = openat(newdfd, tmp, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,
                     st.st_mode & 07777);
        if (out < 0 && errno != EEXIST) { close(in); return -1; }
    }
    if (out < 0) { close(in); return -1; }

    char buf[1 << 16];
    ssize_t n;
    int err = 0;
    while ((n = read(in, buf, sizeof buf)) > 0) {
        ssize_t off = 0;
        while (off < n) {
            ssize_t w = write(out, buf + off, (size_t)(n - off));
            if (w < 0) { err = 1; goto done; }
            off += w;
        }
    }
    if (n < 0) err = 1;
done:
    fchmod(out, st.st_mode & 07777);
    close(in);
    if (close(out) != 0) err = 1;
    if (err) { unlinkat(newdfd, tmp, 0); return -1; }
    if (real_renameat()(newdfd, tmp, newdfd, new) != 0) {
        unlinkat(newdfd, tmp, 0);
        return -1;
    }
    return 0;
}

int rename(const char *old, const char *new) {
    static rename_fn real = NULL;
    if (!real) real = (rename_fn)dlsym(RTLD_NEXT, "rename");
    int rc = real(old, new);
    if (rc == 0 || errno != EXDEV) return rc;
    if (copy_at(AT_FDCWD, old, AT_FDCWD, new) != 0) { errno = EXDEV; return -1; }
    unlinkat(AT_FDCWD, old, 0);
    return 0;
}

int renameat(int olddfd, const char *old, int newdfd, const char *new) {
    static renameat_fn real = NULL;
    if (!real) real = (renameat_fn)dlsym(RTLD_NEXT, "renameat");
    int rc = real(olddfd, old, newdfd, new);
    if (rc == 0 || errno != EXDEV) return rc;
    if (copy_at(olddfd, old, newdfd, new) != 0) { errno = EXDEV; return -1; }
    unlinkat(olddfd, old, 0);
    return 0;
}

int renameat2(int olddfd, const char *old, int newdfd, const char *new,
              unsigned int flags) {
    static renameat2_fn real = NULL;
    if (!real) real = (renameat2_fn)dlsym(RTLD_NEXT, "renameat2");
    int rc = real(olddfd, old, newdfd, new, flags);
    if (rc == 0 || errno != EXDEV) return rc;
    /* RENAME_EXCHANGE / RENAME_NOREPLACE can't be reproduced by copy; pass
       EXDEV through for any non-zero flags. */
    if (flags != 0) { errno = EXDEV; return -1; }
    if (copy_at(olddfd, old, newdfd, new) != 0) { errno = EXDEV; return -1; }
    unlinkat(olddfd, old, 0);
    return 0;
}

int link(const char *old, const char *new) {
    static link_fn real = NULL;
    if (!real) real = (link_fn)dlsym(RTLD_NEXT, "link");
    int rc = real(old, new);
    if (rc == 0 || errno != EXDEV) return rc;
    if (copy_at(AT_FDCWD, old, AT_FDCWD, new) != 0) { errno = EXDEV; return -1; }
    return 0;
}

int linkat(int olddfd, const char *old, int newdfd, const char *new, int flags) {
    static linkat_fn real = NULL;
    if (!real) real = (linkat_fn)dlsym(RTLD_NEXT, "linkat");
    int rc = real(olddfd, old, newdfd, new, flags);
    if (rc == 0 || errno != EXDEV) return rc;
    if (copy_at(olddfd, old, newdfd, new) != 0) { errno = EXDEV; return -1; }
    return 0;
}
"""

_LLRENAME_SHIM_SRC = "/tmp/llrename_shim.c"
_LLRENAME_SHIM_BIN = "/usr/local/lib/llrename_shim.so"
_LD_SO_PRELOAD = "/etc/ld.so.preload"

# Out-of-tree tool dirs some task images ship (editable installs resolved via
# __editable__*.pth). Mapped RO into the sandbox so sglang/megatron/slime
# imports resolve. llrun silently skips any path that doesn't exist, so keeping
# the slime set as the default is safe across all images; override/extend via
# the ``sandbox_ro_tool_dirs`` kwarg.
_DEFAULT_SANDBOX_RO_TOOL_DIRS: tuple[str, ...] = (
    "/sgl-workspace",
    "/root/Megatron-LM",
    "/root/slime",
)

# $HOME for sandboxed (agent/verifier) commands. The container's real home
# (/root) is outside the Landlock allowlist; /workspace is the symlink to the
# writable per-trial workspace, so pointing HOME here keeps tool caches/config
# writable without exposing anything under /user beyond this trial.
_SANDBOX_HOME = "/workspace"


class CybertronEnvironment(BaseEnvironment):
    """Harbor environment backed by a Cybertron devspace over Teleport SSH."""

    @classmethod
    @override
    def preflight(cls) -> None:
        if shutil.which("cctl") is None:
            raise SystemExit(
                "Cybertron environment requires `cctl` on PATH with an active "
                "login (run `cctl login`)."
            )
        # The SSH transport needs either `tsh` (default) or `ssh` (fallback).
        if shutil.which("tsh") is None and not _CURSOR_TSH_FALLBACK.is_file():
            if shutil.which("ssh") is None:
                raise SystemExit(
                    "Cybertron environment requires `tsh` (default transport) "
                    "or `ssh` (with a Teleport Host entry in ~/.ssh/config, "
                    "use_tsh=False) on PATH."
                )

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: TrialPaths,
        task_env_config: EnvironmentConfig,
        *,
        project: str | None = None,
        cluster: str = _DEFAULT_CLUSTER,
        resource_pool: str = _DEFAULT_RESOURCE_POOL,
        billing_account_id: str = _DEFAULT_BILLING_ACCOUNT_ID,
        gpu_model: str = _DEFAULT_GPU_MODEL,
        priority: str = _DEFAULT_PRIORITY,
        requeue: bool = False,
        cpu: int | None = None,
        memory_gb: int | None = None,
        image_id: str | None = None,
        image_map: dict[str, str] | None = None,
        image_map_path: str | Path | None = None,
        cluster_user: str | None = None,
        devspace_id: str | None = None,
        keep_workspace: bool = True,
        cache_dir: str | None = None,
        runs_dir: str | None = None,
        status_timeout_sec: int = _DEFAULT_STATUS_TIMEOUT_SEC,
        ssh_connect_timeout_sec: int = _DEFAULT_SSH_CONNECT_TIMEOUT_SEC,
        heartbeat_interval_sec: int = _DEFAULT_HEARTBEAT_INTERVAL_SEC,
        use_tsh: bool = True,
        tsh_path: str | None = None,
        teleport_cluster: str = _DEFAULT_TELEPORT_CLUSTER,
        ssh_host: str | None = None,
        ssh_config_path: str | Path | None = None,
        cctl_profile: str | None = None,
        sandbox_ro_tool_dirs: list[str] | tuple[str, ...] | None = None,
        **kwargs: Any,
    ) -> None:
        self._project = project
        self._cluster = cluster
        self._resource_pool = resource_pool
        self._billing_account_id = billing_account_id
        self._gpu_model = gpu_model
        # Normalize + validate the scheduling tier up front so a typo fails at
        # construction with a clear message rather than as an opaque cctl error.
        self._priority = str(priority).upper()
        if self._priority not in _VALID_PRIORITIES:
            raise ValueError(
                f"priority must be one of {_VALID_PRIORITIES}, got {priority!r}"
            )
        self._requeue = requeue
        # Explicit cpu/memory_gb win; otherwise scale with the GPU count so the
        # devspace is proportioned to the job (see _CPU_PER_GPU note above). A
        # 0-GPU task is sized as one GPU so it never lands at zero.
        gpu_units = max(task_env_config.gpus or 0, 1)
        self._cpu = cpu if cpu is not None else _CPU_PER_GPU * gpu_units
        self._memory_gb = (
            memory_gb if memory_gb is not None else _MEMORY_GB_PER_GPU * gpu_units
        )
        self._image_id_override = image_id
        # Precedence (low -> high): built-in < image_map_path file < image_map.
        self._image_map = {
            **_DEFAULT_IMAGE_MAP,
            **(_load_image_map_file(image_map_path) if image_map_path else {}),
            **(image_map or {}),
        }
        self._cluster_user_override = cluster_user
        self._keep_workspace = keep_workspace
        self._cache_dir_override = cache_dir
        self._runs_dir_override = runs_dir
        self._status_timeout_sec = status_timeout_sec
        self._ssh_connect_timeout_sec = ssh_connect_timeout_sec
        self._heartbeat_interval_sec = heartbeat_interval_sec

        # SSH transport. Default: `tsh ssh` (needs no ~/.ssh/config Host block,
        # so N concurrent devspaces work unattended). Fallback: plain `ssh
        # <alias>` reading ssh_config_path (the alias must already be present,
        # e.g. written by Cursor's Cybertron plugin).
        self._use_tsh = use_tsh
        self._tsh_path = tsh_path or shutil.which("tsh")
        if self._use_tsh and self._tsh_path is None and _CURSOR_TSH_FALLBACK.is_file():
            self._tsh_path = str(_CURSOR_TSH_FALLBACK)
        self._teleport_cluster = teleport_cluster
        self._ssh_host_override = ssh_host
        self._ssh_config_path = (
            Path(ssh_config_path).expanduser()
            if ssh_config_path is not None
            else Path.home() / ".ssh" / "config"
        )
        self._cctl_profile = cctl_profile
        self._sandbox_ro_tool_dirs: tuple[str, ...] = (
            tuple(sandbox_ro_tool_dirs)
            if sandbox_ro_tool_dirs is not None
            else _DEFAULT_SANDBOX_RO_TOOL_DIRS
        )

        # attach mode: reuse an existing devspace instead of creating one
        self._devspace_id: str | None = devspace_id
        self._owns_devspace = devspace_id is None

        # Resolved Teleport node / ssh alias (same string in both transports).
        self._host: str | None = None
        self._cluster_user: str | None = None
        # Devspace node's primary cluster IP, resolved in start(); used to build
        # the no_proxy bypass so internal cluster traffic skips the HTTP proxy.
        self._node_ip: str | None = None
        self._heartbeat_thread: threading.Thread | None = None
        self._heartbeat_stop = threading.Event()

        super().__init__(
            environment_dir=environment_dir,
            environment_name=environment_name,
            session_id=session_id,
            trial_paths=trial_paths,
            task_env_config=task_env_config,
            **kwargs,
        )

    # -- identity / capabilities -------------------------------------------

    @staticmethod
    @override
    def type() -> EnvironmentType:
        return EnvironmentType.CYBERTRON

    @classmethod
    @override
    def resource_capabilities(cls) -> EnvironmentResourceCapabilities:
        return EnvironmentResourceCapabilities(cpu_limit=True, memory_limit=True)

    @property
    @override
    def capabilities(self) -> EnvironmentCapabilities:
        # mounted=False: all files move via tar-over-ssh (like ACK/EC2).
        # disable_internet=False: devspace has network; agent needs LLM APIs.
        return EnvironmentCapabilities(gpus=True, mounted=False)

    @override
    def _validate_definition(self) -> None:
        if not self.task_env_config.docker_image and self._image_id_override is None:
            raise ValueError(
                "Cybertron environment requires either [environment].docker_image "
                "(mapped to a platform image id) or an explicit image_id kwarg."
            )
        # Fail fast if the image cannot be mapped to a platform id.
        self._resolve_image_id()

    def _resolve_image_id(self) -> str:
        """Resolve the platform image id to pass to ``cctl devspace create``.

        Precedence: explicit ``image_id`` kwarg > the image_map lookup
        (built-in < ``image_map_path`` file < ``image_map`` kwarg, merged in
        ``__init__``) > a raw platform id / canonicalName passed through.
        """
        if self._image_id_override is not None:
            return self._image_id_override
        image = self.task_env_config.docker_image
        if image is None:
            raise ValueError("Cybertron environment has no image configured.")
        mapped = self._image_map.get(image)
        if mapped is not None:
            return mapped
        # Allow passing a raw platform id / canonicalName straight through.
        if image.isdigit() or "/" in image or ":" in image:
            return image
        raise ValueError(
            f"Cannot map image {image!r} to a Cybertron platform image id. "
            "Add it to the cybertron image_map kwarg or pass image_id."
        )

    # -- paths --------------------------------------------------------------

    @property
    def _user(self) -> str:
        if self._cluster_user is None:
            raise RuntimeError("Cluster user has not been resolved yet.")
        return self._cluster_user

    @property
    def _cache_dir(self) -> str:
        return self._cache_dir_override or f"/user/{self._user}/ptdbench-cache"

    @property
    def _run_dir(self) -> str:
        base = self._runs_dir_override or f"/user/{self._user}/harbor-runs"
        safe_session = "".join(
            ch if ch.isalnum() or ch in "-._" else "-" for ch in self.session_id
        )
        return f"{base}/{safe_session}"

    @property
    def _workspace_dir(self) -> str:
        return f"{self._run_dir}/workspace"

    @property
    def _logs_dir(self) -> str:
        return f"{self._run_dir}/logs"

    # -- cctl ---------------------------------------------------------------

    def _cctl_base(self) -> list[str]:
        cmd = ["cctl"]
        if self._cctl_profile:
            cmd += ["--profile", self._cctl_profile]
        return cmd

    async def _cctl(
        self, args: list[str], *, timeout_sec: int = 120
    ) -> tuple[dict[str, Any] | list[Any] | None, int, str]:
        process = await asyncio.create_subprocess_exec(
            *self._cctl_base(),
            *args,
            "-o",
            "json",
            "--no-input",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout_sec
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return None, 124, f"cctl {args[0] if args else ''} timed out"
        text = stdout.decode(errors="replace").strip()
        parsed: dict[str, Any] | list[Any] | None = None
        if text:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
        return parsed, process.returncode or 0, stderr.decode(errors="replace")

    @staticmethod
    def _numeric_id(devspace_id: str) -> str:
        return devspace_id.rsplit("/", 1)[-1]

    async def _create_devspace(self) -> str:
        args = [
            "devspace",
            "create",
            "--project",
            self._require_project(),
            "--cluster",
            self._cluster,
            "--resource-pool",
            self._resource_pool,
            "--billing-account-id",
            self._billing_account_id,
            "--image",
            self._resolve_image_id(),
            "--gpu",
            str(self._effective_gpus),
            "--gpu-model",
            self._gpu_model,
            "--cpu",
            str(self._cpu),
            "--memory",
            str(self._memory_gb),
            "--priority",
            self._priority,
            "--description",
            f"harbor:{self.environment_name}:{self.session_id}",
        ]
        if self._requeue:
            args.append("--requeue")
        parsed, code, stderr = await self._cctl(args, timeout_sec=180)
        if code != 0 or not isinstance(parsed, dict):
            raise RuntimeError(f"cctl devspace create failed: {stderr or parsed}")
        devspace_id = parsed.get("name") or parsed.get("id")
        if not devspace_id:
            raise RuntimeError(f"cctl devspace create returned no id: {parsed}")
        self.logger.info(f"Created Cybertron devspace {devspace_id}")
        return str(devspace_id)

    def _require_project(self) -> str:
        if not self._project:
            raise ValueError(
                "Cybertron environment requires a 'project' kwarg "
                "(e.g. [environment].kwargs.project = 'xinfer-chenyinghao')."
            )
        return self._project

    async def _wait_for_status(self) -> None:
        """Wait until the devspace is Running *and* its ssh host resolves.

        ``cctl`` reports ``Running`` before Teleport registers the node in
        ``tsh ls`` (or before the ssh-config ``Host`` entry is written), so a
        single host-resolution attempt right after ``Running`` can miss a node
        that is genuinely coming up. Both conditions share this one deadline:
        the status poll gates on ``Running``, then host resolution is retried
        until it returns a host. ``_resolve_devspace_host`` returns ``None``
        while the node is still registering (retry) and raises on hard errors
        (ambiguous match, ``tsh ls`` / login failure, missing ssh config), which
        propagate immediately. On success ``self._host`` is set.
        """
        if self._devspace_id is None:
            raise RuntimeError("devspace id is not set")
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._status_timeout_sec
        last = ""
        running = False
        while loop.time() < deadline:
            if not running:
                parsed, code, stderr = await self._cctl(
                    ["devspace", "get", self._devspace_id, "-F", "status"]
                )
                status = ""
                if isinstance(parsed, dict):
                    status = str(parsed.get("status", ""))
                last = status or stderr
                if status in _TERMINAL_STATUSES:
                    raise RuntimeError(
                        f"Devspace {self._devspace_id} entered terminal "
                        f"status {status}."
                    )
                running = status in _RUNNING_STATUSES
                if not running:
                    await asyncio.sleep(10)
                    continue
            # Running: wait for the Teleport node / ssh host to register too.
            host = await self._resolve_devspace_host()
            if host is not None:
                self._host = host
                return
            last = f"Running but ssh host not registered yet (id {self._devspace_id})"
            await asyncio.sleep(10)
        raise RuntimeError(
            f"Devspace {self._devspace_id} did not become ready within "
            f"{self._status_timeout_sec}s (last status: {last})."
        )

    # -- SSH host resolution ------------------------------------------------

    async def _resolve_devspace_host(self) -> str | None:
        """Resolve the Teleport node name / ssh alias for this devspace.

        Order: explicit ``ssh_host`` override; then, in tsh mode, the Teleport
        node whose name ends in ``-<numeric id>`` (``tsh ls``); otherwise the
        matching ``Host`` entry in ``~/.ssh/config``. The same string is used
        as the ssh alias (fallback) and as the ``tsh ssh root@<node>`` target.

        Returns ``None`` when the node/host is not registered *yet* (a transient
        condition ``_wait_for_status`` retries); hard errors (ambiguous match,
        ``tsh ls`` / login failure, missing ssh config) raise immediately.
        """
        if self._ssh_host_override:
            return self._ssh_host_override
        if self._use_tsh:
            return await self._resolve_node_via_tsh()
        return self._resolve_ssh_host()

    async def _resolve_node_via_tsh(self) -> str | None:
        if self._devspace_id is None:
            raise RuntimeError("devspace id is not set")
        if self._tsh_path is None:
            raise RuntimeError(
                "tsh transport selected but no `tsh` binary found "
                "(pass tsh_path, put tsh on PATH, or set use_tsh=False)."
            )
        numeric = self._numeric_id(self._devspace_id)
        process = await asyncio.create_subprocess_exec(
            self._tsh_path,
            "ls",
            "--format",
            "json",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr, code = await self._collect_process(process, timeout_sec=60)
        if code != 0:
            raise RuntimeError(
                f"`tsh ls` failed ({code}); is the Teleport login current "
                f"(`tsh login`)? {stderr.decode(errors='replace')}"
            )
        try:
            nodes = json.loads(stdout.decode(errors="replace") or "[]")
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Could not parse `tsh ls --format json`: {exc}"
            ) from exc
        matches = sorted(
            {
                name
                for node in nodes
                if isinstance(node, dict)
                for name in (
                    node.get("spec", {}).get("hostname")
                    or node.get("metadata", {}).get("name"),
                )
                if isinstance(name, str) and name.endswith(f"-{numeric}")
            }
        )
        if not matches:
            # Not registered in Teleport yet -- cctl reports Running before the
            # node joins `tsh ls`. Transient: let _wait_for_status poll again.
            return None
        if len(matches) > 1:
            raise RuntimeError(
                f"Multiple Teleport nodes match devspace {numeric}: {matches}. "
                "Pass the full node via the ssh_host kwarg."
            )
        return matches[0]

    def _resolve_ssh_host(self) -> str | None:
        """Find the ~/.ssh/config Host entry for this devspace id.

        Teleport writes ``Host devspace-<user>-<proj>-...-<id>`` entries; we
        pick the unique one ending in ``-<numeric id>``. Returns ``None`` when
        no entry matches *yet* (transient: the config may not be refreshed until
        the devspace registers); a missing config file or an ambiguous match
        raises.
        """
        if self._devspace_id is None:
            raise RuntimeError("devspace id is not set")
        numeric = self._numeric_id(self._devspace_id)
        if not self._ssh_config_path.is_file():
            raise RuntimeError(
                f"SSH config {self._ssh_config_path} not found; refresh the "
                "Cybertron/Teleport SSH config (e.g. via Cursor)."
            )
        matches: list[str] = []
        for line in self._ssh_config_path.read_text(errors="replace").splitlines():
            stripped = line.strip()
            if not stripped.lower().startswith("host "):
                continue
            for alias in stripped.split()[1:]:
                if alias.startswith("devspace-") and alias.endswith(f"-{numeric}"):
                    matches.append(alias)
        unique = sorted(set(matches))
        if not unique:
            # Entry not written yet (config refreshed on registration).
            # Transient: let _wait_for_status poll again.
            return None
        if len(unique) > 1:
            raise RuntimeError(
                f"Multiple SSH hosts match devspace {numeric}: {unique}. "
                "Pass the full host via the devspace_id kwarg."
            )
        return unique[0]

    def _resolve_cluster_user(self) -> str:
        # 1) explicit override; 2) parse from Teleport host devspace-<user>-...;
        # 3) fall back to the known account. Never use `whoami` (pod is root).
        if self._cluster_user_override:
            return self._cluster_user_override
        if self._host and self._host.startswith("devspace-"):
            parts = self._host.split("-")
            if len(parts) >= 2 and parts[1]:
                return parts[1]
        return _DEFAULT_CLUSTER_USER

    # -- SSH primitives -----------------------------------------------------

    def _transport_cmd(self) -> list[str]:
        """Argv prefix that runs a remote command; the command is appended.

        tsh mode: ``tsh ssh --cluster=<teleport> root@<node>`` (auth via tsh's
        own key store, no ~/.ssh/config needed). ssh mode: ``ssh <opts>
        <alias>`` with connection multiplexing keyed on the host (``%C``), so
        concurrent devspaces get independent control sockets.
        """
        if self._host is None:
            raise RuntimeError("Devspace SSH host is not resolved yet.")
        if self._use_tsh:
            if self._tsh_path is None:
                raise RuntimeError("tsh transport selected but no `tsh` binary found.")
            return [
                self._tsh_path,
                "ssh",
                f"--cluster={self._teleport_cluster}",
                f"root@{self._host}",
            ]
        return [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={min(30, max(1, self._ssh_connect_timeout_sec))}",
            "-o",
            "ControlMaster=auto",
            "-o",
            "ControlPersist=10m",
            "-o",
            "ControlPath=~/.ssh/devspace-%C",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=3",
            self._host,
        ]

    @staticmethod
    async def _collect_process(
        process: asyncio.subprocess.Process,
        *,
        timeout_sec: int | None,
        stdin: bytes | None = None,
    ) -> tuple[bytes, bytes, int]:
        try:
            if timeout_sec is None:
                stdout, stderr = await process.communicate(stdin)
            else:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(stdin), timeout=timeout_sec
                )
        except asyncio.TimeoutError:
            if process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
            return b"", f"Command timed out after {timeout_sec} seconds".encode(), 124
        return stdout or b"", stderr or b"", process.returncode or 0

    async def _ssh_raw(
        self,
        command: str,
        *,
        stdin: bytes | None = None,
        timeout_sec: int | None = None,
    ) -> tuple[bytes, bytes, int]:
        process = await asyncio.create_subprocess_exec(
            *self._transport_cmd(),
            command,
            # No inherited stdin when we're not piping bytes: tsh ssh has no
            # `-n`, and inheriting a TTY can wedge concurrent transfers.
            stdin=asyncio.subprocess.PIPE
            if stdin is not None
            else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        return await self._collect_process(
            process, timeout_sec=timeout_sec, stdin=stdin
        )

    async def _ssh_exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
        *,
        sandbox: bool = False,
    ) -> ExecResult:
        script = command
        if sandbox:
            script = self._wrap_sandbox(script)
        if env:
            assignments = " ".join(
                f"{key}={shlex.quote(value)}" for key, value in env.items()
            )
            script = f"{assignments} {script}"
        if cwd:
            script = f"cd {shlex.quote(cwd)} && {script}"
        if user is not None and str(user) not in ("root", "0"):
            script = (
                f"runuser -u {shlex.quote(str(user))} -- bash -lc {shlex.quote(script)}"
            )
            remote_command = script
        else:
            remote_command = f"bash -lc {shlex.quote(script)}"

        stdout, stderr, return_code = await self._ssh_raw(
            remote_command, timeout_sec=timeout_sec
        )
        stdout_text = stdout.decode(errors="replace") if stdout else None
        stderr_text = stderr.decode(errors="replace") if stderr else None
        callback = self._output_callback()
        if callback is not None:
            if stdout_text:
                await callback(stdout_text, "stdout")
            if stderr_text:
                await callback(stderr_text, "stderr")
        return ExecResult(
            stdout=stdout_text, stderr=stderr_text, return_code=return_code
        )

    async def _wait_for_ssh(self) -> None:
        self.logger.debug(f"Waiting for SSH on devspace {self._devspace_id}")
        attempts = max(1, self._ssh_connect_timeout_sec // 5)
        last_error = ""
        for _ in range(attempts):
            result = await self._ssh_exec("true", timeout_sec=15)
            if result.return_code == 0:
                return
            last_error = result.stderr or result.stdout or ""
            await asyncio.sleep(5)
        raise RuntimeError(
            f"SSH did not become ready on devspace {self._devspace_id} "
            f"(host {self._host}). Last error: {last_error}"
        )

    # -- proxy bypass -------------------------------------------------------

    def _proxy_preamble(self) -> str:
        """Shell preamble that adds the devspace's own cluster IP to no_proxy.

        Cybertron devspaces export ``http(s)_proxy=whitelist-proxy…:7891``. Any
        request to an *internal* cluster IP (the node's own IP, the ray
        dashboard on localhost, sibling pods) goes through that proxy, which
        502s on internal addresses — e.g. slime's ``_wait_server_healthy``
        polls the sglang engine at the node IP and hangs forever (GPUs at 0%).

        We prepend ``127.0.0.1,localhost,0.0.0.0,<node IP>,<A.B.0.0/16>`` (the
        /16 derived from the node IP so it adapts per cluster) to no_proxy, and
        keep any profile-provided ``$no_proxy`` so *external* hosts still use
        the proxy. Emitted as an ``export`` inside the sandbox's inner
        ``bash -lc`` so it runs *after* the login shell sources the profile and
        therefore wins; children inherit it. Empty until the node IP is known.
        """
        if not self._node_ip:
            return ""
        additions = ["127.0.0.1", "localhost", "0.0.0.0", self._node_ip]
        octets = self._node_ip.split(".")
        if len(octets) == 4:
            additions.append(f"{octets[0]}.{octets[1]}.0.0/16")
        joined = ",".join(additions)
        value = f"{joined}${{no_proxy:+,$no_proxy}}"
        return f'export no_proxy="{value}"; export NO_PROXY="{value}"; '

    async def _resolve_node_ip(self) -> str | None:
        """Best-effort primary cluster IP of the devspace node (for no_proxy).

        The UDP-connect trick returns the interface IP toward the private range
        (the cluster IP, not the proxy); ``hostname -I`` is the fallback. Never
        fatal — on failure the no_proxy bypass is simply skipped.
        """
        probe = (
            'python3 -c "import socket;'
            "s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);"
            "s.connect(('10.255.255.255',1));"
            'print(s.getsockname()[0]);s.close()" 2>/dev/null '
            "|| hostname -I 2>/dev/null | awk '{print $1}'"
        )
        try:
            result = await self._ssh_exec(probe, timeout_sec=30)
        except Exception as exc:  # resolution is best-effort
            self.logger.debug(f"Cybertron node-IP probe failed: {exc}")
            return None
        candidate = (result.stdout or "").strip().split()
        ip = candidate[0] if candidate else ""
        parts = ip.split(".")
        if len(parts) == 4 and all(p.isdigit() and 0 <= int(p) <= 255 for p in parts):
            return ip
        return None

    # -- sandbox (Landlock) -------------------------------------------------

    def _wrap_sandbox(self, command: str) -> str:
        """Wrap *command* so it runs under Landlock via llrun.

        Allowlist (resolved real paths; llrun skips any that don't exist):

        * **RW** — the trial workspace, the trial logs dir (agent logs +
          verifier ``reward.txt``), and the harbor test/solution dirs (test
          scripts often write scratch beside themselves).
        * **RO** — the shared model/dataset cache, ``/harbor`` (skills), and the
          image's out-of-tree tool dirs (``sandbox_ro_tool_dirs``: sglang /
          Megatron / slime editable installs live outside site-packages).

        Everything else under ``/user`` (other users, your other trials) is
        denied. System dirs, ``/dev`` (GPU nodes), ``/tmp`` and ``/dev/shm``
        are hard-coded in llrun. NCCL/CUDA need only those plus the driver
        libs under ``/usr`` — all already covered.

        The inner command is prefixed with :meth:`_proxy_preamble` so internal
        cluster traffic (sglang health checks, ray dashboard) bypasses the
        whitelist HTTP proxy.
        """
        llrun = [
            shlex.quote(_LLRUN_BIN),
            "--rw",
            shlex.quote(self._workspace_dir),
            "--rw",
            shlex.quote(self._logs_dir),
            # Benchmark-authored test/solution trees, on the container rootfs.
            "--rw",
            "/tests",
            "--rw",
            "/solution",
            "--ro",
            shlex.quote(self._cache_dir),
            "--ro",
            "/harbor",
        ]
        for tool_dir in self._sandbox_ro_tool_dirs:
            llrun += ["--ro", shlex.quote(tool_dir)]
        llrun += [
            "--",
            "bash",
            "-lc",
            shlex.quote(self._proxy_preamble() + command),
        ]
        return " ".join(llrun)

    # -- heartbeat ----------------------------------------------------------

    def _start_heartbeat(self) -> None:
        # GPU devspaces are heartbeat-managed: without periodic keepalive the
        # platform snapshots and reclaims them at their deadline. cctl has no
        # heartbeat subcommand, so we keep the devspace warm with a periodic
        # `cctl devspace get` + SSH touch. NOTE(plan §10.1): whether this is
        # sufficient to prevent reclamation must be validated against a long
        # run; swap in the real control-plane :heartbeat endpoint if not.
        if self._heartbeat_interval_sec <= 0:
            return
        self._heartbeat_stop.clear()
        thread = threading.Thread(
            target=self._heartbeat_loop, name="cybertron-heartbeat", daemon=True
        )
        thread.start()
        self._heartbeat_thread = thread

    def _heartbeat_loop(self) -> None:
        while not self._heartbeat_stop.wait(self._heartbeat_interval_sec):
            devspace_id = self._devspace_id
            if devspace_id is None:
                continue
            try:
                subprocess_run = __import__("subprocess").run
                subprocess_run(
                    [
                        *self._cctl_base(),
                        "devspace",
                        "get",
                        devspace_id,
                        "-F",
                        "status",
                    ],
                    capture_output=True,
                    timeout=60,
                )
                if self._host is not None:
                    subprocess_run(
                        [*self._transport_cmd(), "true"],
                        capture_output=True,
                        stdin=__import__("subprocess").DEVNULL,
                        timeout=30,
                    )
            except Exception as exc:  # keepalive is best-effort
                self.logger.debug(f"Cybertron heartbeat error: {exc}")

    def _stop_heartbeat(self) -> None:
        self._heartbeat_stop.set()
        thread = self._heartbeat_thread
        if thread is not None:
            thread.join(timeout=5)
            self._heartbeat_thread = None

    # -- setup --------------------------------------------------------------

    async def _setup_devspace(self) -> None:
        # Build the per-trial dirs on the shared fs, symlink them to the paths
        # ptdbench expects, and compile the Landlock launcher. Runs unconfined.
        setup = f"""
set -e
mkdir -p {shlex.quote(self._workspace_dir)} {shlex.quote(self._logs_dir)}
mkdir -p {shlex.quote(self._cache_dir + "/models")} {shlex.quote(self._cache_dir + "/datasets")}
chmod 777 {shlex.quote(self._logs_dir)}
mkdir -p /ptdbench
# The eval images bake /workspace (WORKDIR) and /ptdbench/{{models,datasets}}
# (RUN mkdir) as real directories. `ln -sfn TARGET DIR` against a real directory
# does NOT replace it: GNU ln silently creates the link *inside* it
# (/workspace/workspace -> ...) and exits 0, so training then writes to the real
# /workspace on the ephemeral overlay instead of the persistent /user target and
# the whole run is lost when the devspace stops. Remove any pre-existing real
# dir first (it is never a correct symlink), then link with -T so any future
# real-dir surprise is a loud failure rather than another silent nest.
for _link in /workspace /logs /ptdbench/models /ptdbench/datasets; do
  if [ -e "$_link" ] && [ ! -L "$_link" ]; then rm -rf "$_link"; fi
done
ln -sfnT {shlex.quote(self._workspace_dir)} /workspace
ln -sfnT {shlex.quote(self._logs_dir)} /logs
ln -sfnT {shlex.quote(self._cache_dir + "/models")} /ptdbench/models
ln -sfnT {shlex.quote(self._cache_dir + "/datasets")} /ptdbench/datasets
# Always (re)compile so the binary tracks the embedded source (cheap, ~1s).
if command -v gcc >/dev/null 2>&1; then
  cat > {shlex.quote(_LLRUN_SRC)} <<'LLRUN_EOF'
{_LLRUN_C}
LLRUN_EOF
  gcc -O2 -o {shlex.quote(_LLRUN_BIN)} {shlex.quote(_LLRUN_SRC)} \
    || echo "llrun: build failed (sandbox disabled)" >&2
elif ! command -v {shlex.quote(_LLRUN_BIN)} >/dev/null 2>&1; then
  echo "llrun: gcc not found and no prebuilt llrun (sandbox disabled)" >&2
fi
# Landlock ABI-v1 rename shim, installed system-wide via /etc/ld.so.preload so
# it composes with any LD_PRELOAD (slime overwrites LD_PRELOAD with
# torch_memory_saver on the train actor). Only written after a successful build
# so a broken .so never lands in ld.so.preload. Idempotent.
if command -v gcc >/dev/null 2>&1; then
  mkdir -p "$(dirname {shlex.quote(_LLRENAME_SHIM_BIN)})"
  cat > {shlex.quote(_LLRENAME_SHIM_SRC)} <<'LLSHIM_EOF'
{_LLRENAME_SHIM_C}
LLSHIM_EOF
  if gcc -O2 -shared -fPIC -o {shlex.quote(_LLRENAME_SHIM_BIN)} \
      {shlex.quote(_LLRENAME_SHIM_SRC)}; then
    grep -qxF {shlex.quote(_LLRENAME_SHIM_BIN)} {shlex.quote(_LD_SO_PRELOAD)} \
      2>/dev/null \
      || echo {shlex.quote(_LLRENAME_SHIM_BIN)} >> {shlex.quote(_LD_SO_PRELOAD)}
  else
    echo "llrename_shim: build failed (cross-dir rename may EXDEV)" >&2
  fi
fi
"""
        result = await self._ssh_exec(setup, timeout_sec=300)
        if result.return_code != 0:
            raise RuntimeError(
                f"Cybertron devspace setup failed: {result.stdout} {result.stderr}"
            )

    # -- lifecycle ----------------------------------------------------------

    @override
    async def start(self, force_build: bool) -> None:
        if self._devspace_id is None:
            self._devspace_id = await self._create_devspace()
        await self._wait_for_status()  # gates on Running + sets self._host
        self._cluster_user = self._resolve_cluster_user()
        self.logger.info(
            f"Devspace {self._devspace_id} ready via SSH host {self._host} "
            f"(user {self._cluster_user})"
        )
        await self._wait_for_ssh()
        self._node_ip = await self._resolve_node_ip()
        if self._node_ip:
            self.logger.debug(f"Devspace node IP {self._node_ip} (no_proxy bypass)")
        self._start_heartbeat()
        await self._setup_devspace()

    @override
    async def stop(self, delete: bool) -> None:
        self._stop_heartbeat()

        if not self._keep_workspace and self._host is not None:
            try:
                await self._ssh_exec(
                    f"rm -rf {shlex.quote(self._run_dir)}", timeout_sec=60
                )
            except Exception as exc:
                self.logger.warning(f"Failed to remove per-trial dir: {exc}")

        if delete and self._owns_devspace and self._devspace_id is not None:
            parsed, code, stderr = await self._cctl(
                ["devspace", "stop", self._devspace_id]
            )
            if code != 0:
                self.logger.warning(
                    f"cctl devspace stop failed for {self._devspace_id}: "
                    f"{stderr or parsed}"
                )
        elif not delete:
            self.logger.warning(
                f"Leaving devspace {self._devspace_id} running (delete=False)."
            )

    # -- exec / transfer ----------------------------------------------------

    @override
    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
        *,
        sandbox: bool = False,
    ) -> ExecResult:
        sandboxed = sandbox or self._in_sandbox_scope()
        merged_env = self._merge_env(env)
        if sandboxed and not (merged_env and "HOME" in merged_env):
            # Landlock denies the container's real $HOME (/root); point HOME at
            # the writable workspace so agent/verifier tools (~/.cache, ~/.config,
            # HF_HOME under $HOME, …) don't hit EACCES.
            merged_env = {**(merged_env or {}), "HOME": _SANDBOX_HOME}
        return await self._ssh_exec(
            command,
            cwd=cwd or self.task_env_config.workdir,
            env=merged_env,
            timeout_sec=timeout_sec,
            user=self._resolve_user(user),
            sandbox=sandboxed,
        )

    @override
    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        source = Path(source_path)
        target = PurePosixPath(target_path)
        target_dir = str(target.parent)
        import io
        import tarfile

        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
            tar.add(source, arcname=target.name)
        remote_command = (
            f"mkdir -p {shlex.quote(target_dir)} && "
            f"tar xf - -C {shlex.quote(target_dir)}"
        )
        _, stderr, return_code = await self._ssh_raw(
            f"bash -lc {shlex.quote(remote_command)}",
            stdin=tar_buffer.getvalue(),
            timeout_sec=300,
        )
        if return_code != 0:
            raise RuntimeError(
                f"Failed to upload {source} to devspace {target_path}: "
                f"{stderr.decode(errors='replace')}"
            )

    @override
    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        source = Path(source_dir)
        if not source.is_dir():
            self.logger.warning(f"No files to upload from {source}")
            return
        tar_buffer = pack_dir_to_bytes(source)
        remote_command = (
            f"mkdir -p {shlex.quote(target_dir)} && "
            f"tar xf - -C {shlex.quote(target_dir)}"
        )
        _, stderr, return_code = await self._ssh_raw(
            f"bash -lc {shlex.quote(remote_command)}",
            stdin=tar_buffer.getvalue(),
            timeout_sec=1800,
        )
        if return_code != 0:
            raise RuntimeError(
                f"Failed to upload directory {source} to devspace {target_dir}: "
                f"{stderr.decode(errors='replace')}"
            )

    @override
    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        import io
        import tarfile

        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        host = PurePosixPath(source_path)
        remote_command = (
            f"tar cf - -C {shlex.quote(str(host.parent))} {shlex.quote(host.name)}"
        )
        stdout, stderr, return_code = await self._ssh_raw(
            f"bash -lc {shlex.quote(remote_command)}", timeout_sec=300
        )
        if return_code != 0:
            raise RuntimeError(
                f"Failed to download devspace file {source_path}: "
                f"{stderr.decode(errors='replace')}"
            )
        tar_buffer = io.BytesIO(stdout)
        with tarfile.open(fileobj=tar_buffer, mode="r") as tar:
            member = next(iter(tar.getmembers()), None)
            if member is None:
                raise RuntimeError(f"Devspace file archive was empty: {source_path}")
            member.name = target.name
            tar.extract(member, path=target.parent, filter="data")

    @override
    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        remote_command = f"cd {shlex.quote(source_dir)} && tar cf - ."
        stdout, stderr, return_code = await self._ssh_raw(
            f"bash -lc {shlex.quote(remote_command)}", timeout_sec=1800
        )
        if return_code != 0:
            raise RuntimeError(
                f"Failed to download devspace directory {source_dir}: "
                f"{stderr.decode(errors='replace')}"
            )
        extract_dir_from_bytes(stdout, target)
