# > | `ptdbench-verl-vllm017` | `25607123` | 
# > | `ptdbench-slime-rc0` | `25607122` | 
# > | `ptdbench-slime-v0.3.2-cu129` | `25607131` |

python3 -m ptdbench harbor-stage tasks/slime_search_r1 --backend cybertron \
  --gpus 8 --image 25607131 \
  --environment-kwarg project=xinfer-chenyinghao \
  --environment-kwarg cluster=beijing \
  --environment-kwarg resource_pool=model-autosft \
  --environment-kwarg billing_account_id=N00069 \
  --environment-kwarg gpu_model=h800 \
  --environment-kwarg priority=NORMAL \
  --run 


#   --devspace-id tasks/1109045 \