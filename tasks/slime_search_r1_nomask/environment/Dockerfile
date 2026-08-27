# Search-R1 historical-run diagnosis on the public slime image.
# Issue grading is log/code review. Solution reruns still need host-side
# Qwen2.5-3B weights, nq_hotpotqa parquet, and the wiki-18 FAISS index.
FROM slimerl/slime:latest

USER root

LABEL org.ptdbench.task="slime_search_r1_nomask" \
      org.ptdbench.framework="slime" \
      org.ptdbench.recipe="examples/search-r1"

RUN python3 -c "import yaml" 2>/dev/null || pip install --no-cache-dir pyyaml tensordict chardet faiss-cpu

COPY hydrate-assets.sh /usr/local/libexec/ptdbench-hydrate-assets.sh
RUN sed -i 's/\r$//' /usr/local/libexec/ptdbench-hydrate-assets.sh \
    && chmod 0755 /usr/local/libexec/ptdbench-hydrate-assets.sh \
    && printf '\n. /usr/local/libexec/ptdbench-hydrate-assets.sh\n' >> /etc/profile

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    WANDB_MODE=offline \
    WANDB_API_KEY=ptdbench-offline \
    WANDB_SILENT=true \
    TOKENIZERS_PARALLELISM=false

WORKDIR /workspace
CMD ["tail", "-f", "/dev/null"]
