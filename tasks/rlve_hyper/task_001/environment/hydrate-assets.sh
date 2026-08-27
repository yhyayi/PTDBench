#!/usr/bin/env sh

if [ "${PTDBENCH_STAGE:-}" = "agent" ] \
    && [ -d /opt/ptdbench-assets/workspace ] \
    && [ ! -e /workspace/.ptdbench-assets-hydrated ]; then
    cp -a /opt/ptdbench-assets/workspace/. /workspace/
    touch /workspace/.ptdbench-assets-hydrated
fi
