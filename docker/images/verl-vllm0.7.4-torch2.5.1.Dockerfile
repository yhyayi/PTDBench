# PTDBench environment `verl-vllm0.7.4-torch2.5.1`.
# ptdbench-image: ptdbench/verl-vllm0.7.4-torch2.5.1:cu124-py3.10
#
# Pinned verl 0.3.0.post1 execution environment: the upstream `hiyouga/verl`
# image at the exact digest the task's defective and repaired runs were recorded
# on. Stack: torch 2.5.1+cu124, vLLM 0.7.4, ray 2.40.0, FSDP, transformers
# (vendored verl pin) -- see environment/source_manifest.json for the vendored
# source and sanitization record. No task payload, model, or dataset is baked in.
# Tasks: infrastructure/verl_checkpoint_host_memory_oom.
#
# PTDBench build-images only pulls; build manually from the docker/ context:
#   docker build -f docker/images/verl-vllm0.7.4-torch2.5.1.Dockerfile docker/
# and tag as the header declares. PTDBench does not bake an activate hook (the
# harness injects one at run time) nor the fetch helper (it ships per task in
# environment/); this is a plain pinned stack image.
FROM hiyouga/verl@sha256:864124d32147f5e9b31916656b441f19d77c1c03e3448c8504b0025df59121b2

ENV CUDA_DEVICE_ORDER=PCI_BUS_ID \
    TOKENIZERS_PARALLELISM=false

# The harness bind-mounts a per-run copy of the task's workspace here.
WORKDIR /workspace
