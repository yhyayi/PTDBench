#!/usr/bin/env sh

if [ "${PTDBENCH_STAGE:-}" = "agent" ] \
    && [ -d /opt/ptdbench-assets/workspace ]; then
    cp -a /opt/ptdbench-assets/workspace/. /workspace/
fi
