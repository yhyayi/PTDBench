FROM node:20-bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        git \
        python3 \
        ripgrep \
    && rm -rf /var/lib/apt/lists/* \
    && npm install --global --prefix /opt/harness-npm \
        @anthropic-ai/claude-code@2.0.55 \
        @openai/codex@0.79.0 \
        @google/gemini-cli@0.18.4 \
        opencode-ai@1.1.59 \
    && npm cache clean --force \
    && rm -rf /root/.npm

ENV PATH="/opt/harness-npm/bin:/usr/local/bin:${PATH}"
WORKDIR /workspace
