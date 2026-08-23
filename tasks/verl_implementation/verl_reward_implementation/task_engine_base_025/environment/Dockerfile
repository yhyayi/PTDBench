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

ARG DATASET_REVISION=ec9d2e88d7729dd73c81715750e8a34ad9718c49
ARG DATASET_URL=https://hf-mirror.com/datasets/LIF1014/ptdbench-verl-reward-implementation-task-engine-base-025-dataset/resolve/${DATASET_REVISION}/artifacts/workspace-assets.tar.gz
ARG DATASET_ARCHIVE_SHA256=7211ce9e85dae7bb431d951ab4970706e29113d06f9e5e6ec5f24c64a7fcf449
ARG DATASET_ARCHIVE_SIZE=2490115

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
