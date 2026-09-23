#!/usr/bin/env bash
# Mirror the git repositories the image build needs onto the host.
#
# Run this on the host (not inside the image build), the same way
# docker/fetch-cuda-wheels.sh is run. Each env-* stage COPYs its environment's
# mirror directory out of the build context, and docker/build-env.sh installs
# from those instead of cloning from github.
#
#   ./docker/fetch-git-mirrors.sh
#   GITHUB_HTTP_PROXY=http://127.0.0.1:8444 ./docker/fetch-git-mirrors.sh
#   GITHUB_GIT_SSH=1 ./docker/fetch-git-mirrors.sh
#
# ---------------------------------------------------------------------------
# Why the fetching moved out of the build
# ---------------------------------------------------------------------------
# Three reasons, in ascending order of how much they cost when ignored:
#
#   1. It is a cache, and the one the build actually wanted. A mirror is
#      cloned once and reused by every rebuild forever after; the pins are
#      commit SHAs, so a mirror that has the commit is done. The pip and conda
#      `--mount=type=cache` mounts that used to sit on the env-* stages cached
#      nothing at all (see docker/Dockerfile) -- this is what they should have
#      been.
#
#   2. Retries belong where a human can see them. build-env.sh used to clone
#      inside the stage with six attempts and a 10s-20s-30s backoff, because a
#      github connect failure there costs 133 seconds of kernel timeout and two
#      in a row was the observed normal. That loop is still here, but a failure
#      now costs you a re-run of a script that takes seconds, not the loss of
#      an hour-deep build stage.
#
#   3. SSH. GITHUB_GIT_SSH exists because github.com:443 is often filtered
#      while :22 is not. Getting an agent into a RUN needs --mount=type=ssh,
#      and on the host it needs nothing: ssh just works, with the user's own
#      agent and keys. It also sidesteps the conda-openssl collision that the
#      in-image version had to work around -- conda-forge ships openssl 3.6,
#      Ubuntu 22.04's /usr/bin/ssh is built against 3.0, and mixing them dies
#      with "OpenSSL version mismatch". No conda prefix exists here, so there
#      is nothing to collide with.
#
# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
#   docker/gitmirror/_store/<safe>.git   cloned once, shared between envs
#   docker/gitmirror/<env>/<safe>.git    hardlink copy of the above
#   docker/gitmirror/<env>/manifest.txt  "<url>\t<dir>", read by build-env.sh
#
# The per-env split matches docker/wheels and exists for the same reason: each
# env-* stage COPYs only its own directory, and a COPY hashes just the subtree
# it names, so bumping slime-v0.3.1's Megatron pin does not invalidate
# slime-rc0's stage. _store is what keeps that from costing a second clone --
# apex is pinned to the same commit by both slime environments, and `cp -al`
# copies it into each env directory for free. _store itself is excluded in
# docker/.dockerignore; no stage COPYs it.
#
# The manifest is the interface. build-env.sh does not re-derive mirror
# directory names from requirements.txt; it reads what this script says it
# wrote. Submodule directory names in particular are not worth deriving twice.
#
# A partial run is safe, which matters because the networks this script exists
# for are the ones that fail halfway. Each entry is appended only after its
# mirror is on disk and verified to contain the pinned commit, so a manifest
# that stops early is still true as far as it goes: build-env.sh installs those
# pins from disk and clones the rest itself. Re-run to pick up where it stopped
# -- a repository already in _store with the right commit is a no-op.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

mirror_root="${repo_root}/docker/gitmirror"
store="${mirror_root}/_store"
mkdir -p "$store"

mapfile -t ENVS < <(find docker/envs -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)

# A filesystem-safe, collision-free directory name for a clone URL.
# https://github.com/NVIDIA/apex.git -> github.com-NVIDIA-apex.git
safe_name() {
    local url="$1"
    url="${url#*://}"
    url="${url%.git}"
    printf '%s.git\n' "$(printf '%s' "$url" | tr -c '[:alnum:]._-' '-')"
}

# github.com:443 filtered but :22 open. Only the transport changes; the URL
# recorded in the manifest, and therefore the one build-env.sh rewrites with
# insteadOf and the one pip resolves, stays https.
maybe_ssh_url() {
    if [ -n "${GITHUB_GIT_SSH:-}" ] && [ "${GITHUB_GIT_SSH}" != "0" ]; then
        case "$1" in
            https://github.com/*) printf 'git@github.com:%s\n' "${1#https://github.com/}"; return ;;
            http://github.com/*)  printf 'git@github.com:%s\n' "${1#http://github.com/}"; return ;;
        esac
    fi
    printf '%s\n' "$1"
}

# GitHub-only proxy, passed per-invocation rather than written to the user's
# ~/.gitconfig. This is someone's actual machine, not a build stage, and a
# script that edits your global git config is a script that has to remember to
# edit it back.
git_args=()
if [ -n "${GITHUB_HTTP_PROXY:-}" ]; then
    for host in github.com codeload.github.com gist.github.com \
                raw.githubusercontent.com objects.githubusercontent.com; do
        git_args+=(-c "http.https://${host}.proxy=${GITHUB_HTTP_PROXY}")
    done
fi

git_retry() {
    local attempt=1
    while :; do
        if git "${git_args[@]}" "$@"; then return 0; fi
        if [ "$attempt" -ge 6 ]; then
            echo "--- gave up after ${attempt} attempts: git $*" >&2
            return 1
        fi
        echo "--- attempt ${attempt} failed, retrying: git $*" >&2
        sleep $((attempt * 10))
        attempt=$((attempt + 1))
    done
}

# Mirror <url> into the shared store and make sure <commit> is present.
# Returns the store directory name on stdout.
#
# Three states, and the middle one is the whole point of this script: the
# commit is already here (do nothing), the repository is here but the pin
# moved (fetch), or nothing is here (clone).
store_mirror() {                    # <url> <commit>
    local url="$1" commit="$2"
    local dir name
    name="$(safe_name "$url")"
    dir="${store}/${name}"

    if [ -d "$dir" ] && git -C "$dir" cat-file -e "${commit}^{commit}" 2>/dev/null; then
        printf '%s\n' "$name"
        return 0
    fi

    if [ -d "$dir" ]; then
        echo "--- fetch ${name} (has no ${commit:0:12} yet)" >&2
        git_retry -C "$dir" fetch --prune origin '+refs/heads/*:refs/heads/*' >&2 || true
    else
        echo "--- clone ${name} <- $(maybe_ssh_url "$url")" >&2
        # Bare, and full rather than --depth 1. pip does not simply clone and
        # check out a git requirement: if anything goes wrong it falls back to
        # `git fetch origin <sha>`, and a fetch against a shallow source is
        # refused -- "shallow roots are not allowed to be updated", once per
        # remote ref. That failure mode was learned the hard way; do not add
        # --depth here to make this script faster.
        rm -rf "${dir}.part"
        # `|| return 1` rather than leaving it to `set -e`: errexit does not
        # reach in here. store_mirror runs inside a command substitution (it
        # returns the directory name on stdout), and a function that fails
        # there does not abort the shell -- without this the clone failure fell
        # through to the mv below and reported itself as "cannot stat
        # .part", which says nothing about github being unreachable.
        git_retry clone --quiet --bare "$(maybe_ssh_url "$url")" "${dir}.part" >&2 \
            || return 1
        # origin is the ssh URL when GITHUB_GIT_SSH rewrote it, which is what
        # the fetch branch above wants on a later run.
        mv "${dir}.part" "$dir"
    fi

    git -C "$dir" cat-file -e "${commit}^{commit}" 2>/dev/null || {
        echo "error: ${url} has no commit ${commit}" >&2
        return 1
    }
    printf '%s\n' "$name"
}

# Submodules recorded by <commit>, first level only -- a nested submodule stays
# pip's problem, and none of the pins here has one. Mirrored because pip runs
# `git submodule update --init --recursive` inside its own clone and one flaky
# fetch of one submodule aborts the whole install twenty minutes in.
# transformer_engine's 3rdparty/googletest is where that actually happened.
mirror_submodules() {               # <store dir name> <commit> <env>
    local name="$1" commit="$2" env_name="$3"
    local dir="${store}/${name}"
    local key url path sha sub
    while read -r key url; do
        [ -n "${key:-}" ] || continue
        case "$url" in
            https://*|http://*) ;;
            *) echo "--- submodule ${url}: not http(s), left to pip"; continue ;;
        esac
        path="$(git -C "$dir" config --blob "${commit}:.gitmodules" \
                    --get "${key%.url}.path")" || continue
        sha="$(git -C "$dir" ls-tree "$commit" "$path" | awk '$2 == "commit" { print $3 }')"
        [ -n "$sha" ] || continue
        echo "--- submodule ${path} <- ${url}@${sha:0:12}"
        sub="$(store_mirror "$url" "$sha")"
        publish "$sub" "$env_name" "$url" "$sha"
    done < <(git -C "$dir" config --blob "${commit}:.gitmodules" \
                 --get-regexp '^submodule\..*\.url$' || true)
}

# Hardlink the store copy into one environment's directory and record
# it in that environment's manifest. `cp -al` rather than `cp -a`: same
# filesystem, so the objects cost nothing twice, and a bare mirror is never
# written to after this point.
#
# Refreshed when the copy is missing the commit being published, which is the
# precise condition rather than a proxy for it: store_mirror fetches into
# _store, and a pin that moved leaves the hardlink copy one commit behind.
# Comparing HEAD instead would miss a pin that moved on a non-default branch.
publish() {                         # <store dir name> <env> <url> <commit>
    local name="$1" env_name="$2" url="$3" commit="$4"
    local dest="${mirror_root}/${env_name}/${name}"
    if [ -d "$dest" ] && ! git -C "$dest" cat-file -e "${commit}^{commit}" 2>/dev/null; then
        rm -rf "$dest"
    fi
    [ -d "$dest" ] || cp -al "${store}/${name}" "$dest"
    printf '%s\t%s\n' "$url" "$name" >> "${mirror_root}/${env_name}/manifest.txt"
}

total=0
for env_name in "${ENVS[@]}"; do
    env_dir="docker/envs/${env_name}"
    out="${mirror_root}/${env_name}"
    mkdir -p "$out"
    # Rewritten from scratch each run: a pin that was removed from
    # requirements.txt must not linger in the manifest and keep an insteadOf
    # alive for a URL nothing asks for any more.
    : > "${out}/manifest.txt"

    echo "=== ${env_name}"
    count=0

    # git+ requirements. Same parse as docker/build-env.sh: "<name> @ git+<url>@<sha>".
    while read -r line; do
        case "$line" in *@*git+*) ;; *) continue ;; esac
        spec="${line#*git+}"
        spec="${spec%%#*}"                      # drop #subdirectory=/#egg=
        spec="${spec%"${spec##*[![:space:]]}"}"
        url="${spec%@*}"
        rev="${spec##*@}"
        name="${line%%[[:space:]]*}"
        name="${name%%[*}"
        case "$url" in https://*|http://*) ;; *) continue ;; esac
        if ! printf '%s' "$rev" | grep -Eq '^[0-9a-fA-F]{40}$'; then
            echo "--- ${name}: ${rev} is not a full commit, left to pip"
            continue
        fi
        echo "--- ${name} <- ${url}@${rev:0:12}"
        stored="$(store_mirror "$url" "$rev")"
        publish "$stored" "$env_name" "$url" "$rev"
        mirror_submodules "$stored" "$rev" "$env_name"
        count=$((count + 1))
    done < <(sed -e 's/#.*//' "${env_dir}/requirements.txt" | awk 'NF')

    # Editable source trees. build-env.sh clones these into the prefix; with an
    # insteadOf pointing at the mirror that clone is local and offline.
    if [ -f "${env_dir}/sources.txt" ]; then
        while read -r dir url commit; do
            [ -n "${dir:-}" ] || continue
            echo "--- ${dir} <- ${url}@${commit:0:12}"
            stored="$(store_mirror "$url" "$commit")"
            publish "$stored" "$env_name" "$url" "$commit"
            count=$((count + 1))
        done < <(sed -e 's/#.*//' "${env_dir}/sources.txt" | awk 'NF')
    fi

    [ "$count" -gt 0 ] || echo "--- no git requirements"
    total=$((total + count))
done

echo
echo "mirrored ${total} repository pin(s) for ${#ENVS[@]} environment(s)"
du -sh "$mirror_root" 2>/dev/null | sed 's/^/total on disk: /'
echo "each env-* stage COPYs docker/gitmirror/<env> out of the build context."
