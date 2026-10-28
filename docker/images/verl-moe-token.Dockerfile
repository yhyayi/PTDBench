# PTDBench environment `verl-moe-token`.
# ptdbench-image: ptdbench/verl-moe-token:vllm0.17-torch2.10-cu129-mcore0.16
#
# The shared `verl` image (same upstream digest) plus one Megatron-Core router
# patch. Stack: torch 2.10.0+cu129, vLLM 0.17.0, megatron-core 0.16.0, mbridge
# 0.15.1, transformer-engine 2.12.0, transformers 4.57.6, ray 2.54.0 -- see
# docker/verl-moe-token/image.json for the image the task was recorded on.
# Tasks: infrastructure/verl_moe_token_normalization.
#
# The patch (docker/verl-moe-token/mcore5047-minimal.patch) backports MCore's
# tp_cp_group scaling of the MoE aux/z-loss in router.py. The recorded runs were
# produced with it applied; it is shared by the defective and repaired arms and
# is not the defect under diagnosis. Verified against megatron-core 0.16.0 in
# the upstream image: `patch -p1` applies cleanly and the result hashes to
# `router_sha256` in image.json.
FROM verlai/verl:vllm017.latest@sha256:4c43bbf17e90284b1102008399240b25406e8d34fea178d86272231b333b7cb6

COPY verl-moe-token/mcore5047-minimal.patch /tmp/mcore5047-minimal.patch
RUN set -eux; \
    mcore="$(python3 -c 'import megatron.core, os; print(os.path.dirname(os.path.dirname(megatron.core.__file__)))')"; \
    patch -d "$mcore" -p1 --forward < /tmp/mcore5047-minimal.patch; \
    rm /tmp/mcore5047-minimal.patch; \
    sha256sum "$mcore/megatron/core/transformer/moe/router.py"; \
    python3 -c "import megatron.core; print('megatron-core', megatron.core.__version__)"
