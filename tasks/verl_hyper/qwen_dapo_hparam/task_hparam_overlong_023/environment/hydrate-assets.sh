#!/bin/sh

# PTDBench bind-mounts the staged workspace over /workspace. Hydrate the
# revision-pinned dataset and checkpoint exactly once per staged workspace.
if [ "${PTDBENCH_STAGE:-}" = "agent" ] \
    && [ -d /opt/ptdbench-assets/workspace ] \
    && [ ! -e /workspace/.ptdbench-assets-hydrated ]; then
    cp -a /opt/ptdbench-assets/workspace/. /workspace/
    touch /workspace/.ptdbench-assets-hydrated
fi
