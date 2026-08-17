FROM verlai/verl:vllm017.latest

USER root

ARG DATASET_REVISION=784a6b8c4f7270e05bed48d1a7cd109abcfa7cc2
ARG DATASET_URL=https://huggingface.co/datasets/LIF1014/ptdbench-qwen-dapo-hparam-task-hparam-seqlen-micro-014-dataset/resolve/${DATASET_REVISION}/artifacts/workspace-assets.tar.gz
ARG DATASET_ARCHIVE_SHA256=e6ff5139b9fd4a4c9cca8d29d1d50faaea2220098e9d97829a737fb50fd314db
ARG DATASET_ARCHIVE_SIZE=5676512

RUN set -eux; \
    if command -v apt-get >/dev/null 2>&1; then \
        apt-get update; \
        apt-get install -y --no-install-recommends ca-certificates curl; \
    fi; \
    mkdir -p /opt/ptdbench-assets; \
    curl --fail --location --retry 5 --retry-all-errors \
        "$DATASET_URL" -o /tmp/workspace-assets.tar.gz; \
    echo "$DATASET_ARCHIVE_SHA256  /tmp/workspace-assets.tar.gz" | sha256sum -c -; \
    test "$(stat -c %s /tmp/workspace-assets.tar.gz)" = "$DATASET_ARCHIVE_SIZE"; \
    tar -xzf /tmp/workspace-assets.tar.gz -C /opt/ptdbench-assets; \
    rm -f /tmp/workspace-assets.tar.gz; \
    if command -v apt-get >/dev/null 2>&1; then rm -rf /var/lib/apt/lists/*; fi

COPY hydrate-assets.sh /usr/local/libexec/ptdbench-hydrate-assets.sh
RUN sed -i 's/\r$//' /usr/local/libexec/ptdbench-hydrate-assets.sh \
    && chmod 0755 /usr/local/libexec/ptdbench-hydrate-assets.sh \
    && printf '\n. /usr/local/libexec/ptdbench-hydrate-assets.sh\n' >> /etc/profile

WORKDIR /workspace
