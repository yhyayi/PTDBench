FROM verlai/verl:vllm017.latest

USER root

ARG MODEL_REPOSITORY=LIF1014/ptdbench-Qwen2.5-0.5B-Instruct
ARG MODEL_REVISION=e967ddaecb746b9cea032fc1ae80c29a9a49bb73
ARG MODEL_WEIGHT_SHA256=fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe
LABEL org.ptdbench.model.repository="${MODEL_REPOSITORY}" \
      org.ptdbench.model.revision="${MODEL_REVISION}" \
      org.ptdbench.model.weight-sha256="${MODEL_WEIGHT_SHA256}"

RUN set -eux; \
    model_dir=/opt/ptdbench-assets/workspace/verl/Qwen2.5-0.5B-Instruct; \
    mkdir -p "${model_dir}"; \
    HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 python3 -c \
      "from huggingface_hub import snapshot_download; snapshot_download(repo_id='${MODEL_REPOSITORY}', revision='${MODEL_REVISION}', local_dir='${model_dir}', allow_patterns=['config.json','generation_config.json','LICENSE','merges.txt','README.md','tokenizer_config.json','tokenizer.json','vocab.json','model.safetensors'], max_workers=1)"; \
    test "$(stat -c %s "${model_dir}/model.safetensors")" = 988097824; \
    echo "${MODEL_WEIGHT_SHA256}  ${model_dir}/model.safetensors" | sha256sum -c -; \
    rm -rf "${model_dir}/.cache"; \
    chmod -R a+rX "${model_dir}"

ARG DATASET_REVISION=ab385ea0b4fd9a8c16680917660d442121a6b5eb
ARG DATASET_BASE_URL=https://hf-mirror.com/datasets/LIF1014/ptdbench-task-hparam-overlong-023-dataset/resolve/ab385ea0b4fd9a8c16680917660d442121a6b5eb/artifacts
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

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    WANDB_MODE=offline \
    WANDB_API_KEY=ptdbench-offline \
    WANDB_SILENT=true \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUDA_VISIBLE_DEVICES=0,1 \
    NVIDIA_VISIBLE_DEVICES=0,1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /workspace
RUN mkdir -p /models /hidden
CMD ["tail", "-f", "/dev/null"]
