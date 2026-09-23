#!/usr/bin/env bash
# Download the CUDA extension wheels the image build wants onto the host.
#
# Run this on the host (not inside the image build). Cluster Docker builds
# typically cannot reach download.pytorch.org or compile flash_attn from the
# PyPI sdist. docker/build-env.sh installs from docker/wheels first and only
# then falls back (pytorch.org for +cuXXX pins, pip/sdist for everything else).
#
#   ./docker/fetch-cuda-wheels.sh
#   GITHUB_HTTP_PROXY=http://127.0.0.1:8445 ./docker/fetch-cuda-wheels.sh
#
# slime-rc0's flash_attn is not downloaded but derived from the upstream
# image; see derive_rc0_flash_attn below. That step needs docker and, unless
# the image is already local, a registry this host can reach:
#
#   PTDBENCH_SLIME_RC0_IMAGE=docker.m.daocloud.io/slimerl/slime:v0.5.0rc0-cu126 \
#       ./docker/fetch-cuda-wheels.sh
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/wheels"

# One subdirectory per environment, and the split is load-bearing for the build
# cache rather than mere tidiness. Each env-* stage in docker/Dockerfile COPYs
# only docker/wheels/<env>, and a COPY hashes just the subtree it names, so a
# new slime-v0.3.1 torch no longer touches verl's stage. Under the old flat
# directory every wheel's digest landed in all three stages' cache keys, and
# adding one file sent all three back to a full recompile -- hours, to install a
# wheel two of them never look at.
envs=(verl slime-rc0 slime-v0.3.1)
for name in "${envs[@]}"; do
    mkdir -p "${root}/${name}"
done

# cp312 / x86_64: all three conda environments are python 3.12 on linux-64.
# Torch pins are the +cuXXX lines in
# docker/envs/{verl,slime-rc0,slime-v0.3.1}/requirements.txt.
# flash_attn has no CUDA local version in the pin (flash_attn==2.8.3); the
# torch/CUDA tag lives in the wheel filename instead. PyPI only ships an
# sdist, so a missing file here means the image build compiles it.
#
# Each entry is "<env> <url>": the environment whose stage gets the file, and
# where to get it. The environment is written out rather than inferred from the
# CUDA tag because two environments can want the same tag for different reasons,
# and a wheel filed under the wrong one is invisible to the stage that needs it.
#
# This directory is no longer shared between environments, but the torch/CUDA
# tags in the filenames still matter and docker/build-env.sh still checks them:
# they are what makes a wheel self-describing, and `wheel_for_pin` refusing a
# mismatch is the backstop for a file landing in the wrong env directory here.
# Do not drop either tag from a filename to "simplify" it.
torch_base="https://mirrors.aliyun.com/pytorch-wheels"
wheels=(
    # --- verl: torch 2.10, CUDA 12.9 -------------------------------------
    "verl ${torch_base}/cu129/torch-2.10.0+cu129-cp312-cp312-manylinux_2_28_x86_64.whl"
    "verl ${torch_base}/cu129/torchaudio-2.10.0+cu129-cp312-cp312-manylinux_2_28_x86_64.whl"
    "verl ${torch_base}/cu129/torchvision-0.25.0+cu129-cp312-cp312-manylinux_2_28_x86_64.whl"
    # verl: flash_attn==2.8.3, torch 2.10, python 3.12. Official Dao-AILab
    # v2.8.3 has no cu12+torch2.10 wheel; this is the matching prebuild
    # (cu128 binaries run on the env's CUDA 12.9).
    "verl https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.12/flash_attn-2.8.3%2Bcu128torch2.10-cp312-cp312-linux_x86_64.whl"

    # --- slime-rc0: torch 2.8, CUDA 12.6 ---------------------------------
    "slime-rc0 ${torch_base}/cu126/torch-2.8.0+cu126-cp312-cp312-manylinux_2_28_x86_64.whl"
    "slime-rc0 ${torch_base}/cu126/torchaudio-2.8.0+cu126-cp312-cp312-manylinux_2_28_x86_64.whl"
    "slime-rc0 ${torch_base}/cu126/torchvision-0.23.0+cu126-cp312-cp312-manylinux_2_28_x86_64.whl"
    "slime-rc0 ${torch_base}/cu126/torchao-0.9.0+cu126-cp39-abi3-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl"
    # slime-rc0's flash_attn==2.7.4.post1 used to be fetched from a
    # third-party prebuild here. It was built against a pre-2.8 torch and its
    # extension would not load (undefined symbol: _ZN3c104cuda9SetDeviceEa,
    # the 1-arg c10::cuda::SetDevice that torch 2.8 replaced). It comes out of
    # the upstream image now -- same binary slime itself ships. See
    # derive_rc0_flash_attn below.

    # --- slime-v0.3.1: torch 2.11, CUDA 13.0 -----------------------------
    # 13.0, not 12.9 like the other two: sglang 0.5.15.post1 requires it.
    "slime-v0.3.1 ${torch_base}/cu130/torch-2.11.0+cu130-cp312-cp312-manylinux_2_28_x86_64.whl"
    "slime-v0.3.1 ${torch_base}/cu130/torchaudio-2.11.0+cu130-cp312-cp312-manylinux_2_28_x86_64.whl"
    "slime-v0.3.1 ${torch_base}/cu130/torchvision-0.26.0+cu130-cp312-cp312-manylinux_2_28_x86_64.whl"
    # cp310-abi3, not cp312: torchao ships one stable-ABI wheel per CUDA
    # build. It is the right file for 3.12 -- abi3 means "3.10 and newer".
    "slime-v0.3.1 ${torch_base}/cu130/torchao-0.17.0+cu130-cp310-abi3-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl"
    # Same flash_attn version as verl, a different torch *and* a different CUDA
    # major. The +cu128 build of this same wheel links libcudart.so.12 and
    # cannot load here at all; nothing but the tag in this filename says so.
    # They live in separate directories now, which is a second layer of
    # protection on top of the filename tags -- not a replacement for them.
    "slime-v0.3.1 https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.4/flash_attn-2.8.3%2Bcu130torch2.11-cp312-cp312-linux_x86_64.whl"
)

fetch() {
    local env_name="$1"
    local url="$2"
    local dest="${root}/${env_name}"
    local name="${url##*/}"
    # GitHub release URLs encode '+' as %2B; the on-disk name must match
    # what pip and wheel_for_pin look for (flash_attn-2.8.3+cu128...).
    name="${name//%2B/+}"
    name="${name//%2b/+}"
    local path="${dest}/${name}"
    local expected
    local curl_args=( -fL --retry 5 --retry-delay 2 --globoff )
    # Same knob as the image build: GitHub-only, never a global HTTPS_PROXY
    # (that would send the Aliyun/Tsinghua fetches through the tunnel too).
    case "$url" in
        *github.com*|*githubusercontent.com*)
            if [ -n "${GITHUB_HTTP_PROXY:-}" ]; then
                curl_args+=( --proxy "$GITHUB_HTTP_PROXY" )
            fi
            ;;
    esac
    # One-time migration from the flat docker/wheels layout this script used to
    # write. The file itself is unchanged -- only which directory it belongs in
    # is new -- and re-downloading 3 GB to move it would be absurd.
    if [ ! -f "$path" ] && [ -f "${root}/${name}" ]; then
        echo "move  ${name} -> ${env_name}/"
        mv "${root}/${name}" "$path"
    fi
    # A size probe, not a gate: mirrors go down, start refusing HEAD, or move a
    # path, and none of that is a reason to discard a wheel that is already
    # here and complete. `|| true` because pipefail would otherwise make the
    # probe's failure the script's.
    expected="$(curl -sSI "${curl_args[@]}" "$url" 2>/dev/null | awk 'BEGIN{IGNORECASE=1} /^content-length:/{print $2}' | tr -d '\r' | tail -n1 || true)"
    if [ -f "$path" ]; then
        if [ -z "${expected:-}" ]; then
            local host="${url#*://}"
            echo "skip  ${name} (present; ${host%%/*} did not answer the size probe)"
            return 0
        fi
        if [ "$(stat -c%s "$path")" = "$expected" ]; then
            echo "skip  ${name} (already ${expected} bytes)"
            return 0
        fi
    fi
    echo "get   ${name}"
    curl "${curl_args[@]}" -C - -o "${path}.part" "$url"
    mv "${path}.part" "$path"
    echo "done  ${name} ($(stat -c%s "$path") bytes)"
}

for entry in "${wheels[@]}"; do
    read -r entry_env entry_url <<<"$entry"
    fetch "$entry_env" "$entry_url"
done

# ---------------------------------------------------------------------------
# slime-rc0's flash_attn: taken from the upstream image, not from a prebuild site
# ---------------------------------------------------------------------------
# Dao-AILab's official v2.7.4.post1 stops at torch 2.7, so someone else's build
# has to fill the gap -- and a flash_attn binary is only valid against the exact
# torch C++ ABI it was compiled against. slimerl/slime built its own, left the
# wheel in its pip cache, and that is the binary slime's recorded outcomes were
# produced with. Taking that one removes the guesswork entirely.
#
# Only slime-rc0 needs this. slime-v0.3.1 pins flash_attn==2.8.3, which the
# mjun0812 prebuild set covers for torch 2.11 on CUDA 13.0 (+cu130torch2.11) --
# it is in `wheels` above.
#
# It is repacked under a local version (+cu126torch2.8) rather than merely
# renamed: pip rejects a wheel whose filename version disagrees with its
# METADATA. `flash_attn==2.7.4.post1` still matches it -- PEP 440 says a
# specifier with no local segment matches any local version -- and the tag is
# then visible in `pip freeze` and in docker/wheels, which is the whole point.
rc0_flash_attn="flash_attn-2.7.4.post1+cu126torch2.8-cp312-cp312-linux_x86_64.whl"
rc0_image="${PTDBENCH_SLIME_RC0_IMAGE:-${PTDBENCH_SLIME_IMAGE:-slimerl/slime:v0.5.0rc0-cu126}}"

derive_rc0_flash_attn() {
    local dest="${root}/slime-rc0"
    local path="${dest}/${rc0_flash_attn}"
    if [ ! -f "$path" ] && [ -f "${root}/${rc0_flash_attn}" ]; then
        echo "move  ${rc0_flash_attn} -> slime-rc0/"
        mv "${root}/${rc0_flash_attn}" "$path"
    fi
    if [ -f "$path" ]; then
        echo "skip  ${rc0_flash_attn} (already derived)"
        return 0
    fi
    if ! command -v docker >/dev/null 2>&1; then
        # Consistent with every other miss here: absent means the image build
        # compiles the sdist. That is slow (an hour or two) but correct -- it
        # compiles against the env's own torch.
        echo "warn  no docker; ${rc0_flash_attn} not derived" >&2
        echo "      the slime-rc0 stage will compile flash_attn from the PyPI sdist" >&2
        return 0
    fi
    if ! docker image inspect "$rc0_image" >/dev/null 2>&1; then
        echo "pull  ${rc0_image}"
        docker pull "$rc0_image"
    fi

    # Where the wheel is, and which torch it was built against. Two runs rather
    # than one two-line probe: an empty `find` result would otherwise shift the
    # torch version up into the path variable and fail somewhere less obvious.
    local cached torch_version tag
    cached="$(docker run --rm --entrypoint sh "$rc0_image" -c \
        'find /root/.cache/pip/wheels -name "flash_attn-2.7.4.post1-*.whl" -print -quit')"
    case "$cached" in
        /*.whl) ;;
        *)
            echo "error: ${rc0_image} has no cached flash_attn wheel under" >&2
            echo "       /root/.cache/pip/wheels; it is not the image this expects" >&2
            exit 1
            ;;
    esac
    torch_version="$(docker run --rm --entrypoint python3 "$rc0_image" \
        -c 'import torch; print(torch.__version__)')"
    # 2.8.0+cu126 -> cu126torch2.8, the tag the filename above claims.
    tag="${torch_version#*+}torch$(printf '%s' "${torch_version%%+*}" | cut -d. -f1,2)"
    [ "$rc0_flash_attn" = "flash_attn-2.7.4.post1+${tag}-cp312-cp312-linux_x86_64.whl" ] || {
        echo "error: ${rc0_image} now ships torch ${torch_version}, which does not match" >&2
        echo "       ${rc0_flash_attn}. The slime-rc0 pin moved: update this script and" >&2
        echo "       docker/envs/slime-rc0/requirements.txt together." >&2
        exit 1
    }

    echo "derive ${rc0_flash_attn} <- ${rc0_image}:${cached}"
    local cid
    cid="$(docker create "$rc0_image" /bin/true)"
    # docker cp writes as the invoking user; a bind mount would write as root.
    docker cp "${cid}:${cached}" "${path}.orig" >/dev/null
    docker rm "$cid" >/dev/null
    python3 - "${path}.orig" "${path}.part" "$tag" <<'PY'
import base64, hashlib, sys, zipfile

src_path, out_path, tag = sys.argv[1:4]
OLD, NEW = "2.7.4.post1", f"2.7.4.post1+{tag}"
old_di, new_di = f"flash_attn-{OLD}.dist-info/", f"flash_attn-{NEW}.dist-info/"

src = zipfile.ZipFile(src_path)
record = []

# The defect this whole path exists to avoid: torch 2.8 changed
# c10::cuda::SetDevice from one argument to two, and a wheel built against an
# older torch asks the dynamic linker for a symbol that no longer exists. It
# only shows up at `import flash_attn`, long after the build.
so = "flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so"
blob = src.read(so)
if b"_ZN3c104cuda9SetDeviceEab" not in blob:
    sys.exit(f"{src_path}: {so} does not reference the 2-argument "
             "c10::cuda::SetDevice; it was built against a pre-2.8 torch")
del blob

with zipfile.ZipFile(out_path, "w", allowZip64=True) as out:
    for info in src.infolist():
        name = info.filename
        if name == old_di + "RECORD":
            continue                      # regenerated below
        new_name = new_di + name[len(old_di):] if name.startswith(old_di) else name
        ni = zipfile.ZipInfo(new_name, date_time=info.date_time)
        ni.external_attr = info.external_attr      # keeps the .so executable
        ni.compress_type = info.compress_type
        ni.create_system = info.create_system

        if name == old_di + "METADATA":
            text = src.read(name).decode()
            if f"\nVersion: {OLD}\n" not in text:
                sys.exit(f"{src_path}: unexpected METADATA layout")
            data = text.replace(f"\nVersion: {OLD}\n", f"\nVersion: {NEW}\n", 1).encode()
            out.writestr(ni, data)
            record.append((new_name, hashlib.sha256(data).digest(), len(data)))
            continue

        digest, size = hashlib.sha256(), 0
        with src.open(info) as fh, out.open(ni, "w") as wh:
            while chunk := fh.read(1 << 22):
                digest.update(chunk)
                size += len(chunk)
                wh.write(chunk)
        if not new_name.endswith("/"):
            record.append((new_name, digest.digest(), size))

    lines = ["%s,sha256=%s,%d" % (n, base64.urlsafe_b64encode(d).rstrip(b"=").decode(), s)
             for n, d, s in record]
    lines.append(new_di + "RECORD,,")
    # Same timestamp and mode as the RECORD being replaced: writestr() with a
    # bare name would stamp the current time, which is the one thing that would
    # stop two runs of this script from producing the same bytes.
    old_record = src.getinfo(old_di + "RECORD")
    ri = zipfile.ZipInfo(new_di + "RECORD", date_time=old_record.date_time)
    ri.external_attr = old_record.external_attr
    ri.compress_type = old_record.compress_type
    ri.create_system = old_record.create_system
    out.writestr(ri, "\r\n".join(lines) + "\r\n")
PY
    mv "${path}.part" "$path"
    rm -f "${path}.orig"
    echo "done  ${rc0_flash_attn} ($(stat -c%s "$path") bytes)"
}

derive_rc0_flash_attn

# ---------------------------------------------------------------------------
# Every flash_attn wheel here must say which torch it was built against
# ---------------------------------------------------------------------------
# The filename is the only place that information survives -- the metadata
# version of a stock build is a bare 2.7.4.post1 no matter which torch produced
# it. An untagged file is therefore unverifiable by inspection, and the one that
# got here that way was built against the wrong torch and broke `import
# flash_attn` in the finished image. Refuse rather than delete: it may be a
# perfectly good wheel that only needs repacking under its real tag.
untagged=()
shopt -s nullglob
for name in "${envs[@]}"; do
    for path in "${root}/${name}"/flash_attn-*.whl "${root}/${name}"/flash-attn-*.whl; do
        case "${path##*/}" in
            # Both tags, not just torch: a flash_attn binary is pinned to one
            # torch C++ ABI *and* one CUDA major, and docker/build-env.sh
            # checks both.
            *+cu[0-9][0-9]*torch[0-9].[0-9]*) ;;
            *) untagged+=("${name}/${path##*/}") ;;
        esac
    done
done
# Anything still sitting at the top level is a leftover the per-env migration
# above did not claim -- no stage COPYs it and no build will ever see it, which
# is a far quieter failure than an untagged name.
stray=()
for path in "$root"/*.whl; do
    stray+=("${path##*/}")
done
shopt -u nullglob
if ((${#untagged[@]})); then
    echo >&2
    echo "error: flash_attn wheel(s) in ${root} carry no +cu<NNN>torch<X.Y> tag:" >&2
    printf '  %s\n' "${untagged[@]}" >&2
    echo "a flash_attn binary is only valid against the torch it was compiled" >&2
    echo "against and the CUDA major it links, and an untagged filename hides" >&2
    echo "both. Remove it, or repack it as" >&2
    echo "flash_attn-<version>+cu<NNN>torch<X.Y>-...whl once you know (nm -D the" >&2
    echo "bundled .so: _ZN3c104cuda9SetDeviceEab means torch>=2.8, and objdump -p" >&2
    echo "it | grep libcudart tells you the CUDA major)." >&2
    exit 1
fi
if ((${#stray[@]})); then
    echo >&2
    echo "error: wheel(s) directly in ${root}, which no env-* stage reads:" >&2
    printf '  %s\n' "${stray[@]}" >&2
    echo "each stage COPYs docker/wheels/<env>, so a file here is" >&2
    echo "invisible to the build. Move it into the environment that wants it," >&2
    echo "or delete it." >&2
    exit 1
fi

echo
for name in "${envs[@]}"; do
    echo "wheels for ${name}:"
    ls -lh "${root}/${name}"/*.whl 2>/dev/null || echo "  (none)"
done

