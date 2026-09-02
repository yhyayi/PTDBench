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

COPY environment/hydrate-assets.sh /usr/local/libexec/ptdbench-hydrate-assets.sh
RUN sed -i 's/\r$//' /usr/local/libexec/ptdbench-hydrate-assets.sh \
    && chmod 0755 /usr/local/libexec/ptdbench-hydrate-assets.sh \
    && printf '\n. /usr/local/libexec/ptdbench-hydrate-assets.sh\n' >> /etc/profile

# Build context is this task directory (tasks_coding_pass_rate or tasks_coding_pass_01). All assets live under workspace/.
COPY workspace/ /opt/ptdbench-assets/workspace/

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
