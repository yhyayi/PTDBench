# --image 只认 image 名（任务 metadata.yaml 的 image 字段），后端查 _DEFAULT_IMAGE_MAP 映射到 canonicalName：
# > | `verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12` |
# > | `slime-v0.3.2-sglang0.5.15-torch2.11.0-flashattn2.8.3-cu129-py3.12` |
# > | `vllm0.11.0-torch2.8.0-flashattn2.8.3-cu128-py3.12` |

python3 -m ptdbench harbor-stage tasks/slime_search_r1 --backend cybertron \
  --environment-kwarg project=xinfer-chenyinghao \
  --environment-kwarg cluster=beijing \
  --environment-kwarg resource_pool=model-autosft \
  --environment-kwarg billing_account_id=N00069 \
  --environment-kwarg gpu_model=h800 \
  --environment-kwarg priority=NORMAL \
  --run 


#   --devspace-id tasks/1109045 \