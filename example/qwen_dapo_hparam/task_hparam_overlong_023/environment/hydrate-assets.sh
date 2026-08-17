#!/bin/sh

# PTDBench bind-mounts the staged workspace over /workspace, which hides files
# written there while the image is built. Restore the verified, image-local
# assets into that writable mount when the agent container starts.
if [ "${PTDBENCH_STAGE:-}" = "agent" ] \
    && [ -d /opt/ptdbench-assets/workspace ]; then
    cp -a /opt/ptdbench-assets/workspace/. /workspace/
fi
