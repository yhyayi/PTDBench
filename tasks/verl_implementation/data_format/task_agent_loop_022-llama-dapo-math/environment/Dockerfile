FROM verlai/verl:vllm017.latest

USER root

ARG MODEL_REPOSITORY=LIF1014/ptdbench-Llama-3.2-1B-Instruct
ARG MODEL_REVISION=32f9e721f796d3c639ab64d2f58ceeaa2c3213b9
ARG MODEL_WEIGHT_SHA256=1ff795ff6a07e6a68085d206fb84417da2f083f68391c2843cd2b8ac6df8538f
LABEL org.ptdbench.model.repository="${MODEL_REPOSITORY}" \
      org.ptdbench.model.revision="${MODEL_REVISION}" \
      org.ptdbench.model.weight-sha256="${MODEL_WEIGHT_SHA256}"

RUN set -eux; \
    model_dir=/opt/ptdbench-assets/workspace/verl/Llama-3.2-1B-instruct; \
    mkdir -p "${model_dir}"; \
    HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 python3 -c \
      "from huggingface_hub import snapshot_download; snapshot_download(repo_id='${MODEL_REPOSITORY}', revision='${MODEL_REVISION}', local_dir='${model_dir}', allow_patterns=['config.json','generation_config.json','LICENSE.txt','README.md','USE_POLICY.md','special_tokens_map.json','tokenizer_config.json','tokenizer.json','tokenizer.model','model.safetensors'], max_workers=1)"; \
    test "$(stat -c %s "${model_dir}/model.safetensors")" = 2471645608; \
    echo "${MODEL_WEIGHT_SHA256}  ${model_dir}/model.safetensors" | sha256sum -c -; \
    rm -rf "${model_dir}/.cache"; \
    chmod -R a+rX "${model_dir}"

ARG DATASET_REVISION=87e9793734359014fcbc22fa62d866930180b606
ARG DATASET_URL=https://hf-mirror.com/datasets/LIF1014/ptdbench-data-format-task-agent-loop-022-llama-dapo-math-dataset/resolve/${DATASET_REVISION}/artifacts/workspace-assets.tar.gz
ARG DATASET_ARCHIVE_SHA256=6fea5d02c79e23bd1f12085b1638a2e9980623fd1a6fc2446d62497a2ae933de
ARG DATASET_ARCHIVE_SIZE=6174498

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
