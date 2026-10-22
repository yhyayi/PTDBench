FROM verlai/verl:vllm017.latest

USER root

# pyext 0.7 still calls inspect.getargspec, which Python 3.12 removed.
# Fetch the sdist with urllib so pip does not import the unpatched module.
RUN set -eux; \
    python3 -c 'import urllib.request; urllib.request.urlretrieve("https://pypi.tuna.tsinghua.edu.cn/packages/b0/be/9b6005ac644aaef022527ce49617263379e49dbdbd433d1d3dd66d71f570/pyext-0.7.tar.gz", "/tmp/pyext-0.7.tar.gz")'; \
    mkdir -p /tmp/pyext-src; \
    tar -xzf /tmp/pyext-0.7.tar.gz -C /tmp/pyext-src --strip-components=1; \
    sed -i 's/inspect\.getargspec/inspect.getfullargspec/g' /tmp/pyext-src/pyext.py; \
    python3 -m pip install --no-cache-dir --no-build-isolation /tmp/pyext-src; \
    python3 -c 'from pyext import RuntimeModule'; \
    rm -rf /tmp/pyext-src /tmp/pyext-0.7.tar.gz

ARG MODEL_REPOSITORY=LIF1014/ptdbench-verl-implementation-torch-functional
ARG MODEL_REVISION=36a4c226ef70aef2c38df173a7cf86a47ad3ad00
ARG MODEL_WEIGHT_SHA256=dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee
LABEL org.ptdbench.model.repository="${MODEL_REPOSITORY}" \
      org.ptdbench.model.revision="${MODEL_REVISION}" \
      org.ptdbench.model.weight-sha256="${MODEL_WEIGHT_SHA256}"

RUN set -eux; \
    model_dir=/opt/ptdbench-assets/workspace/Qwen2.5-1.5B-Instruct; \
    mkdir -p "${model_dir}"; \
    HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 python3 -c \
      "from huggingface_hub import snapshot_download; snapshot_download(repo_id='${MODEL_REPOSITORY}', revision='${MODEL_REVISION}', local_dir='${model_dir}', allow_patterns=['config.json','generation_config.json','LICENSE','merges.txt','README.md','tokenizer_config.json','tokenizer.json','vocab.json','model.safetensors'], max_workers=1)"; \
    test "$(stat -c %s "${model_dir}/model.safetensors")" = 3087467144; \
    echo "${MODEL_WEIGHT_SHA256}  ${model_dir}/model.safetensors" | sha256sum -c -; \
    rm -rf "${model_dir}/.cache"; \
    chmod -R a+rX "${model_dir}"

ARG DATASET_REVISION=22a1fee0ea2527f514c4115f6c792216a99d7e6f
ARG DATASET_URL=https://hf-mirror.com/datasets/LIF1014/ptdbench-data-task-train-too-hard-dataset/resolve/${DATASET_REVISION}/artifacts/workspace-assets.tar.gz
ARG DATASET_ARCHIVE_SHA256=eddeece8c2828e21b581d5168dd4604470165cb061a53e68d50b26d49ce6d916
ARG DATASET_ARCHIVE_SIZE=167612568

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
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    NVIDIA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /workspace
