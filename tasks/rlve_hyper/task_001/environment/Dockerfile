FROM slimerl/slime:v0.5.0rc0-cu126

USER root

ARG MODEL_REPOSITORY=LIF1014/ptdbench-Qwen2.5-0.5B-Instruct
ARG MODEL_REVISION=e967ddaecb746b9cea032fc1ae80c29a9a49bb73
ARG MODEL_WEIGHT_SHA256=fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe
LABEL org.ptdbench.model.repository="${MODEL_REPOSITORY}" \
      org.ptdbench.model.revision="${MODEL_REVISION}" \
      org.ptdbench.model.weight-sha256="${MODEL_WEIGHT_SHA256}"

# Bake the HF snapshot into /opt/ptdbench-assets. The agent bind-mounts
# /workspace, so hydrate-assets.sh copies these files in at agent start.
# Build this image on a node that can reach hf-mirror (e.g. g91), then
# docker save | docker load onto air-gapped nodes (e.g. g29).
RUN set -eux; \
    model_dir=/opt/ptdbench-assets/workspace/Qwen2.5-0.5B-Instruct; \
    mkdir -p "${model_dir}"; \
    HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 python3 -c \
      "from huggingface_hub import snapshot_download; snapshot_download(repo_id='${MODEL_REPOSITORY}', revision='${MODEL_REVISION}', local_dir='${model_dir}', allow_patterns=['config.json','generation_config.json','LICENSE','merges.txt','README.md','tokenizer_config.json','tokenizer.json','vocab.json','model.safetensors'], max_workers=1)"; \
    test "$(stat -c %s "${model_dir}/model.safetensors")" = 988097824; \
    echo "${MODEL_WEIGHT_SHA256}  ${model_dir}/model.safetensors" | sha256sum -c -; \
    rm -rf "${model_dir}/.cache"; \
    chmod -R a+rX "${model_dir}"

ARG DATASET_REVISION=1c33b31d0632b6a5508b8f1b88d3c98f1cf4e2e7
ARG DATASET_URL=https://hf-mirror.com/datasets/LIF1014/ptdbench-rlve-hyper-task-001-dataset/resolve/${DATASET_REVISION}/artifacts/workspace-assets.tar.gz
ARG DATASET_ARCHIVE_SHA256=d7164e8c27586efae4daeb972f858092390c475938fcb848e213612bb4190a9e
ARG DATASET_ARCHIVE_SIZE=9207578

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
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    NVIDIA_VISIBLE_DEVICES=0,1,2,3 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /workspace
RUN mkdir -p /opt/bench /models /hidden
