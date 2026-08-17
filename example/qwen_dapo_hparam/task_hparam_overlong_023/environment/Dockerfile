FROM verlai/verl:vllm017.latest

ARG DATASET_REVISION=ab385ea0b4fd9a8c16680917660d442121a6b5eb
ARG DATASET_BASE_URL=https://huggingface.co/datasets/LIF1014/ptdbench-task-hparam-overlong-023-dataset/resolve/ab385ea0b4fd9a8c16680917660d442121a6b5eb/artifacts
ARG TRAIN_SHA256=2897263b99e9b4af7523edeb5ec1ab9cac6e1bba63f0bfa78d4cd9356426ca5b
ARG TEST_SHA256=f860d1c3bfb0ce4a937b5e91bb43549a0a7ded51b1725c58c9c10f9f075d2aeb

LABEL org.ptdbench.dataset.repository="LIF1014/ptdbench-task-hparam-overlong-023-dataset" \
      org.ptdbench.dataset.revision="${DATASET_REVISION}"

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends ca-certificates coreutils curl; \
    asset_dir=/opt/ptdbench-assets/workspace/DAPO-Math-17k_verl; \
    encoded_dir=/tmp/ptdbench-dataset; \
    mkdir -p "$asset_dir" "$encoded_dir"; \
    for index in 0 1 2 3 4 5 6 7 8 9; do \
        part="$(printf '%05d' "$index")"; \
        curl --fail --location --retry 5 --retry-all-errors \
            --connect-timeout 15 --max-time 300 \
            "$DATASET_BASE_URL/train.parquet.b64.part-$part.txt" \
            -o "$encoded_dir/train-$part.b64"; \
    done; \
    curl --fail --location --retry 5 --retry-all-errors \
        --connect-timeout 15 --max-time 300 \
        "$DATASET_BASE_URL/test.parquet.b64.txt" \
        -o "$encoded_dir/test.b64"; \
    cat "$encoded_dir"/train-*.b64 | base64 --decode > "$asset_dir/train.parquet"; \
    base64 --decode "$encoded_dir/test.b64" > "$asset_dir/test.parquet"; \
    echo "$TRAIN_SHA256  $asset_dir/train.parquet" | sha256sum --check -; \
    echo "$TEST_SHA256  $asset_dir/test.parquet" | sha256sum --check -; \
    test "$(stat -c %s "$asset_dir/train.parquet")" = 5885886; \
    test "$(stat -c %s "$asset_dir/test.parquet")" = 675271; \
    rm -rf "$encoded_dir" /var/lib/apt/lists/*

COPY hydrate-assets.sh /usr/local/libexec/ptdbench-hydrate-assets.sh
RUN sed -i 's/\r$//' /usr/local/libexec/ptdbench-hydrate-assets.sh \
    && chmod 0755 /usr/local/libexec/ptdbench-hydrate-assets.sh \
    && printf '\n. /usr/local/libexec/ptdbench-hydrate-assets.sh\n' >> /etc/profile

WORKDIR /workspace
RUN mkdir -p /models /hidden
CMD ["tail", "-f", "/dev/null"]
