# 公共参数（每条命令都带）；不加 --run 即 dry-run，只打印将提交的 cctl job 命令
COMMON=(--project xinfer-chenyinghao
        --cluster beijing --resource-pool model-autosft --billing-account-id N00069
        --gpu 1 --gpu-model h800 --cpu 16 --memory 128
        --git-path git@codehub.devcloud.cn-north-4.huaweicloud.com:090558ff0a7d43b1921956dc141bf7f8/ptdbench.git --git-ref cyh
        --cluster-user chenyinghao)

# ① slime-rc0（cu126）: rlve_hyper 四个子任务
python harbor/scripts/cybertron_preheat.py "${COMMON[@]}" \
  --image 25607122 \
  --prepare-glob 'tasks/rlve_hyper/*/environment/prepare.sh' \
  --run

# ② slime-v0.3.2（cu129）: slime_retool + length_hack（search_r1* 见下方说明，暂缓）
python harbor/scripts/cybertron_preheat.py "${COMMON[@]}" \
  --image 25607131 \
  --prepare-glob 'tasks/slime_retool/environment/prepare.sh tasks/slime_retool_length_hack/environment/prepare.sh' \
  --run

# ③ verl（cu129）: reward_design ×2 + verl_hyper ×1 + verl_implementation ×3
python harbor/scripts/cybertron_preheat.py "${COMMON[@]}" \
  --image 25607123 \
  --prepare-glob 'tasks/reward_design/*/environment/prepare.sh tasks/verl_hyper/*/*/environment/prepare.sh tasks/verl_implementation/*/*/environment/prepare.sh' \
  --run