#!/usr/bin/env bash
# Verify a built PTDBench image: gate G2.
#
# Each conda environment must import its own stack, reach the GPU, and carry
# kernels compiled for the cards we target. The build already checked that the
# stacks import at all; everything that needs a driver is here, which is why
# this is separate from the build and needs GPUs.
#
# Everything goes through `sh -lc` with PTDBENCH_STAGE / PTDBENCH_CONDA_ENV set,
# because that is exactly how the harness runs commands. Invoking
# /opt/conda/envs/<name>/bin/python directly would work, and would test a path
# no evaluation ever takes -- in particular it would skip the activate hook,
# which is where the per-stage differences live.
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

image_tag="${PTDBENCH_IMAGE_TAG:-ptdbench/env:latest}"

# Every environment the image is supposed to contain, read from the registry so
# this file does not need editing when one is added.
mapfile -t ENVS < <(find docker/envs -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)

# Pick the emptiest GPU rather than taking all of them. These boxes are shared,
# and a neighbour's training job filling every card would otherwise turn this
# suite into a wall of red that says nothing about the image. Override with
# PTDBENCH_SMOKE_GPU (e.g. "--gpus all", or "" to skip GPU work entirely).
pick_gpu() {
    command -v nvidia-smi >/dev/null 2>&1 || { echo "--gpus all"; return; }
    local best
    best=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
           | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')
    [ -n "$best" ] && echo "--gpus device=$best" || echo "--gpus all"
}
gpu_flag="${PTDBENCH_SMOKE_GPU-$(pick_gpu)}"

failures=0
skipped=0

# A check exiting with this means "the host could not provide what the check
# needs", not "the image is wrong". Kept distinct so a busy GPU never reads as
# a broken stack.
EX_HOST_BUSY=75

run_in() {
    # shellcheck disable=SC2086 # gpu_flag intentionally word-splits
    docker run --rm $gpu_flag "$@"
}

# run_as <env> <stage> <script>  -- the harness's own entry path.
run_as() {
    local env="$1" stage="$2" script="$3"
    run_in --entrypoint sh \
        -e "PTDBENCH_STAGE=$stage" -e "PTDBENCH_CONDA_ENV=$env" \
        "$image_tag" -lc "$script"
}

check() {
    local label="$1"; shift
    local status output
    printf '%-52s' "$label"
    output="$("$@" 2>&1)"; status=$?
    if [ "$status" -eq 0 ]; then
        echo "ok"
        [ -n "${PTDBENCH_SMOKE_VERBOSE:-}" ] && echo "$output" | sed 's/^/    /'
    elif [ "$status" -eq "$EX_HOST_BUSY" ]; then
        echo "SKIP (host)"
        echo "$output" | grep -i 'SKIP:' | sed 's/^/    /'
        skipped=$((skipped + 1))
    else
        echo "FAIL"
        echo "$output" | sed 's/^/    /'
        failures=$((failures + 1))
    fi
    return 0
}

echo "smoke testing $image_tag"
echo "environments: ${ENVS[*]}"
echo

# --- the property the whole design rests on ----------------------------------
# Build prefix == run prefix. If an interpreter, a console script shebang or a
# package file ever resolves outside its own environment prefix, something was
# relocated and the previous design's failure mode is back. This is cheap, needs
# no GPU, and is the single most valuable check in the file.
for env in "${ENVS[@]}"; do
    check "$env: everything resolves inside its own prefix" \
        run_as "$env" agent 'python3 - <<PY
import os, subprocess, sys, sysconfig
prefix = "/opt/conda/envs/" + os.environ["PTDBENCH_CONDA_ENV"]

def must_be_inside(label, path):
    assert path.startswith(prefix + "/"), f"{label} is outside the prefix: {path}"

must_be_inside("sys.executable", sys.executable)
must_be_inside("sys.prefix", sys.prefix + "/")
must_be_inside("purelib", sysconfig.get_paths()["purelib"])

# A console script keeps its shebang from install time. This is the exact
# artifact that could not survive being moved, so it is the one to assert on.
pip = subprocess.run(["sh", "-c", "command -v pip"], capture_output=True, text=True).stdout.strip()
must_be_inside("pip", pip)
with open(pip) as handle:
    shebang = handle.readline().strip()
assert prefix in shebang, f"pip shebang points elsewhere: {shebang}"

import torch
must_be_inside("torch", torch.__file__)
print("prefix", prefix, "torch", torch.__version__)
PY'
done

# --- imports + CUDA + target architectures -----------------------------------
# The allocation is separated from the imports because the two fail for
# unrelated reasons. A failed import means the image is wrong and is worth
# waking someone for; a failed allocation usually means a neighbour is using the
# card. Only the first is a FAIL.
#
# get_arch_list() is the check that TORCH_CUDA_ARCH_LIST actually took. Without
# it, a stack built for the builder's card imports cleanly, reaches the GPU,
# and then fails or silently recompiles at the first kernel launch.
cuda_probe='
import torch
if not torch.cuda.is_available():
    print("SKIP: no GPU visible to this container")
    raise SystemExit(75)
arches = torch.cuda.get_arch_list()
missing = [a for a in ("sm_80", "sm_90") if a not in arches]
assert not missing, f"built without {missing}; have {arches}"
try:
    torch.zeros(1).cuda()
except Exception as exc:
    if "out of memory" in str(exc).lower():
        print("SKIP: imports fine, but no free GPU memory on this host (%s)" % type(exc).__name__)
        raise SystemExit(75)
    raise
print("cuda ok", torch.version.cuda, arches)
'

check "verl: torch + vllm + TE + CUDA" \
    run_as verl agent "python3 - <<'PY'
import torch, vllm, transformer_engine
print('imports ok', torch.__version__, vllm.__version__)
$cuda_probe
PY"

check "slime-rc0: torch + sglang + TE + megatron + CUDA" \
    run_as slime-rc0 agent "python3 - <<'PY'
import torch, sglang, transformer_engine, megatron.core
print('imports ok', torch.__version__, sglang.__version__)
$cuda_probe
PY"

check "slime-v0.3.1: torch + sglang + TE + megatron + CUDA" \
    run_as slime-v0.3.1 agent "python3 - <<'PY'
import torch, sglang, transformer_engine, megatron.core
print('imports ok', torch.__version__, sglang.__version__)
$cuda_probe
PY"

# --- the sglang private symbols each stack needs -----------------------------
# Both stacks import these directly. They are private, so they are exactly the
# kind of thing an unnoticed sglang bump would remove -- and the two stacks do
# NOT need the same set, which is the concrete reason they cannot be merged:
# GPU_MEMORY_TYPE_CUDA_GRAPH does not exist in sglang 0.5.0rc0.
check "slime-rc0: sglang private API rlve uses" \
    run_as slime-rc0 agent 'python3 - <<"PY"
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS
print("ok")
PY'

check "slime-v0.3.1: sglang private API retool/search-r1 use" \
    run_as slime-v0.3.1 agent 'python3 - <<"PY"
from sglang.srt.constants import (
    GPU_MEMORY_TYPE_CUDA_GRAPH,
    GPU_MEMORY_TYPE_KV_CACHE,
    GPU_MEMORY_TYPE_WEIGHTS,
)
print("ok")
PY'

# --- the four things slime reaches for *after* the imports above succeed ------
# Everything here was absent from a slime-rc0 image that passed every check in
# this file, because `import sglang` says nothing about whether the serving
# runtime came with it and `import megatron.core` says nothing about the patches
# the upstream image applied to that tree. Each line below is a failure we have
# actually had, and each one surfaced far from its cause.
#
# No GPU needed: these are imports and a stat. Keeping them GPU-free means they
# still run on a host whose cards are all busy, which is when a missing package
# is most likely to be mistaken for a busy neighbour.
check "slime-rc0: rollout + ref-load entry points" \
    run_as slime-rc0 agent 'python3 - <<"PY"
import os, sysconfig

# 1. The serving runtime. sglang installed without its [srt] extra imports
# perfectly and dies here instead, inside a Ray actor about ten minutes into a
# run: slime/backends/sglang_utils/sglang_engine.py:7 -> No module named uvloop.
from sglang.srt.entrypoints.http_server import launch_server

# 2. The preload shim. slime/ray/actor_group.py asserts this file exists and
# puts it in LD_PRELOAD before starting a training actor; torch_memory_saver
# from PyPI ships the Python package without building it, so `import
# torch_memory_saver` is not evidence that it is there.
so = os.path.join(
    sysconfig.get_paths()["purelib"], "torch_memory_saver_hook_mode_preload.abi3.so"
)
assert os.path.exists(so), f"torch_memory_saver preload hook missing: {so}"

# 3. The first import of the HF -> torch_dist converter, which is what produces
# the --ref-load checkpoint in environment/prepare.sh. It needs the custom_fsdp
# alias from patches/Megatron-LM/0001; without it prepare fails and rlve.sh has
# no fallback guard to cover for it.
from mbridge import AutoBridge

# 4. The offload hooks from patches/Megatron-LM/0002. slime calls these behind
# `hasattr`, so an unpatched megatron makes them silent no-ops -- the run
# proceeds and simply holds NCCL buffers through every rollout. Nothing else in
# this suite would notice.
from megatron.core import parallel_state as mpu
for name in ("destroy_process_groups", "reload_process_groups"):
    assert hasattr(mpu, name), f"megatron not patched: mpu.{name} is missing"

print("ok", os.path.basename(so))
PY'

# --- apex's fused weight-gradient kernel -------------------------------------
# megatron's ColumnParallelLinear/RowParallelLinear default to
# gradient_accumulation_fusion=True and raise at construction if this extension
# is missing. The build already checked that it imports; here it is actually
# called, because an apex built without --cuda_ext can still leave an importable
# pure-Python shim behind. Both slime stacks need it.
apex_probe='
import torch, fused_weight_gradient_mlp_cuda as fw
if not torch.cuda.is_available():
    print("SKIP: no GPU visible to this container")
    raise SystemExit(75)
x = torch.randn(8, 16, dtype=torch.bfloat16, device="cuda")
g = torch.randn(8, 32, dtype=torch.bfloat16, device="cuda")
w = torch.zeros(32, 16, dtype=torch.float32, device="cuda")
fw.wgrad_gemm_accum_fp32(x, g, w)
torch.cuda.synchronize()
assert torch.isfinite(w).all(), "wgrad produced non-finite values"
assert w.abs().sum().item() > 0, "wgrad left the accumulator untouched"
print("apex wgrad ok", fw.__file__)
'

for env in slime-rc0 slime-v0.3.1; do
    check "$env: apex fused wgrad kernel runs" \
        run_as "$env" agent "python3 - <<'PY'
$apex_probe
PY"
done

# --- apex carries both target architectures ----------------------------------
# The probe above only proves apex works on whatever card this host has. That is
# not the same property: an apex compiled for sm_80 alone passes it on an A800
# and then fails at the first kernel launch on an H100. torch has
# get_arch_list() and the cuda_probe checks it; apex's extensions expose no such
# thing, so the cubins are read directly.
#
# This is worth a check of its own because apex is the one package here whose
# pinned commit is *about* architecture flags: slime-rc0 pins
# NVIDIA/apex@26bba57d, "Remove version dependent macros & extension specific
# arch flags except fmha" (see docker/envs/README.md delta 13). A future bump
# that quietly stops honouring TORCH_CUDA_ARCH_LIST would leave every other
# check in this file green.
#
# No GPU needed -- it reads the .so files. apex_C is skipped: it is the
# host-side C++ extension and contains no device code at all.
arch_probe='
import glob, os, subprocess, sys, sysconfig

cuobjdump = os.path.join(sys.prefix, "bin", "cuobjdump")
if not os.path.exists(cuobjdump):
    print("SKIP: no cuobjdump in this prefix (%s)" % cuobjdump)
    raise SystemExit(75)

purelib = sysconfig.get_paths()["purelib"]
# Only the first is required. It is the one megatron cannot start without, and
# the one the probe above already calls. The rest are checked when they are
# there: which extensions apex emits varies with its commit and its CUDA, and a
# name that moved should not read as a missing architecture.
names = ["fused_weight_gradient_mlp_cuda", "amp_C", "fused_layer_norm_cuda",
         "scaled_upper_triang_masked_softmax_cuda"]
found = [(n, so[0]) for n in names for so in [glob.glob(os.path.join(purelib, n + ".*.so"))] if so]
assert found and found[0][0] == names[0], f"apex extension absent: {names[0]}"

for name, so in found:
    listing = subprocess.run([cuobjdump, "--list-elf", so],
                             capture_output=True, text=True).stdout
    have = {p for p in listing.replace(".", " ").split() if p.startswith("sm_")}
    lacks = [a for a in ("sm_80", "sm_90") if a not in have]
    assert not lacks, f"{name} built without {lacks}; have {sorted(have) or None}"
    print(name, sorted(have))
'

for env in slime-rc0 slime-v0.3.1; do
    check "$env: apex cubins cover sm_80 and sm_90" \
        run_as "$env" agent "python3 - <<'PY'
$arch_probe
PY"
done

# --- source trees where the launch scripts look ------------------------------
# slime's scripts *assign* PYTHONPATH=/workspace:/root/Megatron-LM, and repeat
# it inside the Ray runtime_env JSON where an outer export cannot reach. The
# path has to be real, and it has to point at *this* env's tree: the two stacks
# carry different megatron-core versions behind the same absolute path.
for env in slime-rc0 slime-v0.3.1; do
    check "$env: /root/Megatron-LM resolves and imports" \
        run_as "$env" agent '
         test -d /root/Megatron-LM || { echo "missing /root/Megatron-LM"; exit 1; }
         case "$(readlink -f /root/Megatron-LM)" in
            "/opt/conda/envs/$PTDBENCH_CONDA_ENV/src/Megatron-LM") ;;
            *) echo "points at $(readlink -f /root/Megatron-LM), not this env"; exit 1 ;;
         esac
         PYTHONPATH=/root/Megatron-LM python3 -c "import megatron.core as m; print(m.__file__)"'
done

# --- environments do not leak into each other --------------------------------
# The stacks differ by a whole torch major-minor and a CUDA version. A package
# from one imported into another is an extension linking against the wrong
# libtorch: a segfault, or a wrong number, rather than an ImportError.
#
# The two slime environments are checked against each other too, not just
# against verl. They are the pair most likely to leak -- same framework, same
# /root/Megatron-LM path, adjacent prefix names -- and the least likely to say
# so out loud when they do.
leak_probe() {
    # leak_probe <expected-torch-version-prefix>
    printf 'import sys, os, torch\n'
    printf 'mine = "/opt/conda/envs/" + os.environ["PTDBENCH_CONDA_ENV"]\n'
    printf 'leaked = [p for p in sys.path if p.startswith("/opt/conda/envs/") and not p.startswith(mine + "/")]\n'
    printf 'assert not leaked, f"another prefix on sys.path: {leaked}"\n'
    printf 'assert torch.__version__.startswith("%s"), torch.__version__\n' "$1"
    printf 'print("isolated;", torch.__version__)\n'
}

check "verl: no other prefix on the path" \
    run_as verl agent "python3 - <<'PY'
$(leak_probe 2.10.)
PY"

check "slime-rc0: no other prefix on the path" \
    run_as slime-rc0 agent "python3 - <<'PY'
$(leak_probe 2.8.)
PY"

check "slime-v0.3.1: no other prefix on the path" \
    run_as slime-v0.3.1 agent "python3 - <<'PY'
$(leak_probe 2.11.)
PY"

# --- the activate hook's per-stage differences -------------------------------
# The verifier must not inherit anywhere the agent could have installed things:
# grading cannot depend on what the agent did to its own environment.
check "agent stage gets a private user base" \
    run_as verl agent \
    'test "${PIP_USER:-}" = "1" || { echo "PIP_USER=${PIP_USER:-unset}"; exit 1; }
     test -n "${PYTHONUSERBASE:-}" || { echo "PYTHONUSERBASE unset"; exit 1; }
     case "$PYTHONUSERBASE" in /workspace/*) ;; *) echo "$PYTHONUSERBASE"; exit 1 ;; esac
     echo ok'

check "verifier stage does not" \
    run_as verl verifier \
    'test -z "${PIP_USER:-}" || { echo "PIP_USER leaked: $PIP_USER"; exit 1; }
     python3 -c "import torch; print(torch.__version__)"'

# --- prepared/env is data, not shell -----------------------------------------
# The file prepare.sh writes is sourced by every command in the later stages.
# It has to be read as KEY=VALUE assignments; if it were `.`-sourced, a task
# package would be running arbitrary shell in front of the verifier too.
check "prepared/env is read as assignments, not executed" \
    run_in --entrypoint sh \
        -e PTDBENCH_STAGE=agent -e PTDBENCH_CONDA_ENV=verl \
        "$image_tag" -c '
     mkdir -p /ptdbench/prepared
     printf "PTDBENCH_TEST_OK=yes\ntouch /tmp/pwned\nPTDBENCH_TEST_SPACES=a b c\n" \
        > /ptdbench/prepared/env
     sh -lc "
        test \"\$PTDBENCH_TEST_OK\" = yes || { echo \"not set\"; exit 1; }
        test \"\$PTDBENCH_TEST_SPACES\" = \"a b c\" || { echo \"spaces mangled\"; exit 1; }
        test ! -e /tmp/pwned || { echo \"the file was executed as shell\"; exit 1; }
        echo ok
     " 2>/dev/null'

# --- the recorded closure travels with the image -----------------------------
# G1 diffs these against the upstream images freeze. They are useless if they
# did not make it into the final stage.
check "freeze manifests shipped" \
    run_in --entrypoint sh "$image_tag" -c '
     for e in '"${ENVS[*]}"'; do
        test -s "/opt/ptdbench/freeze/$e.txt" || { echo "missing freeze for $e"; exit 1; }
        printf "%s %s packages\n" "$e" "$(wc -l < /opt/ptdbench/freeze/$e.txt)"
     done'

echo
if [ "$failures" -ne 0 ]; then
    echo "$failures check(s) failed"
    exit 1
fi
if [ "$skipped" -ne 0 ]; then
    echo "all checks passed ($skipped skipped: the host had no free GPU)"
    echo "the image is fine; re-run when a card frees up to cover the CUDA path"
    exit 0
fi
echo "all checks passed"
