#!/bin/bash

if [ $# -lt 3 ]; then
    echo "Usage: $0 WANDB_PROJECT RUN_NAME ENVIRONMENT_LIST [EXTRA_TRAIN_ARGS...]"
    exit 1
fi

# BENCH: locked training task types for this workspace — do not change names.
LOCKED_ENVIRONMENT_LIST="MafMafia"

WANDB_PROJECT=$1
RUN_NAME=$2
ENVIRONMENT_LIST="${LOCKED_ENVIRONMENT_LIST}"
ENVIRONMENT_LIST="${LOCKED_ENVIRONMENT_LIST}"
shift 3
EXTRA_ARGS=("$@")

# NOTE: This key is hardcoded at user request for quick launch.
# For better security, prefer exporting WANDB_API_KEY in shell/secret manager instead.
: "${WANDB_API_KEY:?Set WANDB_API_KEY in the runtime environment}"
# Legacy global cleanup mode is now opt-in only.
# Default behavior avoids killing unrelated runs in the same container.
RLVE_ENABLE_GLOBAL_CLEANUP="${RLVE_ENABLE_GLOBAL_CLEANUP:-0}"
echo "RLVE_ENABLE_GLOBAL_CLEANUP=${RLVE_ENABLE_GLOBAL_CLEANUP}"

if [ "${RLVE_ENABLE_GLOBAL_CLEANUP}" = "1" ]; then
    echo "RLVE_ENABLE_GLOBAL_CLEANUP=1: running legacy global cleanup."
    pkill -9 sglang || true
    sleep 3
    ray stop --force || true
    pkill -9 ray || true
    pkill -9 python || true
    sleep 3
    pkill -9 ray || true
    pkill -9 python || true
else
    echo "Skip global cleanup (default=0). Set RLVE_ENABLE_GLOBAL_CLEANUP=1 to enable legacy behavior."
fi

set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16
# fix tokenizers parallelism warning when using multiprocessing
export TOKENIZERS_PARALLELISM=false

NVLINK_COUNT=$(nvidia-smi | grep -o "NVLink" | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

source scripts/models/qwen2.5-0.5B.sh

# Default run storage root is ../experiement; override with RLVE_RUN_ROOT_DIR if needed.
RUN_ROOT_DIR=${RLVE_RUN_ROOT_DIR:-"../experiement"}
RUN_DIR="${RUN_ROOT_DIR%/}/${RUN_NAME}"
mkdir -p "${RUN_DIR}"
echo "RUN_DIR=${RUN_DIR}"

CKPT_ARGS=(
   --hf-checkpoint "${RLVE_HF_CHECKPOINT:-/home/test/test06/lyh/Qwen2.5-0.5B-Instruct}"
   --ref-load "${RLVE_REF_LOAD:-/home/test/test06/lyh/Qwen2.5-0.5B-Instruct_torch_dist}"
   --load "${RUN_DIR}/"
   --save "${RUN_DIR}/"
   --save-interval 20
)

ROLLOUT_ARGS=(
   --disable-rollout-global-dataset
   --rlve
   --environment-list "${ENVIRONMENT_LIST}"

   --custom-prompt-preprocessor ChatTemplate_NoSystemPrompt
   --apply-chat-template

   --rm-type rlve
   --reward-key reward

   --num-rollout 100
   --rollout-batch-size 2
   --n-samples-per-prompt 8
   --rollout-max-response-len 2048
   --rollout-temperature 1.0

   --over-sampling-batch-size 16

   --num-steps-per-rollout 1
   --wandb-always-use-train-step
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data HELD-OUT_ENVIRONMENTS_128 data/HELD-OUT_ENVIRONMENTS/test_128.json
   --n-samples-per-eval-prompt 1
   --eval-top-p 0.7

   --eval-input-key user_prompt
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   # --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 4
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 1024 # Lowered for stability to reduce OOM risk.

   # --optimizer-cpu-offload
   # --overlap-cpu-optimizer-d2h-h2d
   # --use-precision-aware-optimizer
)

GRPO_ARGS=(
   --advantage-estimator grpo
   # --use-kl-loss
   # --kl-loss-coef 0
   # --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28

   --use-tis
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 2e-6
   --lr-decay-style constant
   --weight-decay 0.01
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project "${WANDB_PROJECT}"
   --wandb-group "${RUN_NAME}"
   --wandb-key "${WANDB_API_KEY}"
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.7
   --sglang-server-concurrency 256
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
)

# Per-run log directory (one folder per launch, two files inside).
RUN_TS=$(date +%Y%m%d_%H%M%S)
RUN_LOG_DIR="logs/${RUN_NAME}/run_${RUN_TS}"
LAUNCH_LOG="${RUN_LOG_DIR}/launcher.log"
RAY_JOB_LOG="${RUN_LOG_DIR}/ray_job.log"
mkdir -p "${RUN_LOG_DIR}"

# Mirror script stdout/stderr to launcher log while keeping terminal output.
exec > >(tee -a "${LAUNCH_LOG}") 2>&1
echo "RUN_LOG_DIR=${RUN_LOG_DIR}"
echo "LAUNCH_LOG=${LAUNCH_LOG}"
echo "RAY_JOB_LOG=${RAY_JOB_LOG}"

# Derive visible GPU count from CUDA_VISIBLE_DEVICES for flexible multi-GPU launch.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -ra _VISIBLE_GPU_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
    NUM_GPUS=${#_VISIBLE_GPU_ARRAY[@]}
else
    NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
fi
echo "NUM_GPUS_FOR_TRAINING: ${NUM_GPUS} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-ALL})"
VISIBLE_GPU_SIGNATURE="${CUDA_VISIBLE_DEVICES:-ALL}"
VISIBLE_GPU_SIGNATURE="${VISIBLE_GPU_SIGNATURE// /}"

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
# Ray runtime selection strategy:
# - Use a deterministic "port namespace" (default keyed by visible GPU set) so
#   multiple experiments on the same node can run isolated Ray clusters.
# - You can override any port by setting the corresponding RLVE_RAY_* env var.
RLVE_AUTO_RAY_PORT_BY_GPU="${RLVE_AUTO_RAY_PORT_BY_GPU:-1}"
RLVE_RAY_CLUSTER_KEY="${RLVE_RAY_CLUSTER_KEY:-${VISIBLE_GPU_SIGNATURE}}"
RLVE_RAY_PORT_SPAN="${RLVE_RAY_PORT_SPAN:-100}"

RLVE_RAY_GCS_BASE_PORT="${RLVE_RAY_GCS_BASE_PORT:-6379}"
# Backward-compatible alias: RLVE_RAY_BASE_DASHBOARD_PORT (old name).
RLVE_RAY_DASHBOARD_BASE_PORT="${RLVE_RAY_DASHBOARD_BASE_PORT:-${RLVE_RAY_BASE_DASHBOARD_PORT:-8265}}"
RLVE_RAY_CLIENT_BASE_PORT="${RLVE_RAY_CLIENT_BASE_PORT:-10001}"
RLVE_RAY_METRICS_BASE_PORT="${RLVE_RAY_METRICS_BASE_PORT:-18080}"
RLVE_RAY_NODE_MANAGER_BASE_PORT="${RLVE_RAY_NODE_MANAGER_BASE_PORT:-19000}"
RLVE_RAY_OBJECT_MANAGER_BASE_PORT="${RLVE_RAY_OBJECT_MANAGER_BASE_PORT:-20000}"
RLVE_RAY_DASHBOARD_AGENT_LISTEN_BASE_PORT="${RLVE_RAY_DASHBOARD_AGENT_LISTEN_BASE_PORT:-21000}"
RLVE_RAY_DASHBOARD_AGENT_GRPC_BASE_PORT="${RLVE_RAY_DASHBOARD_AGENT_GRPC_BASE_PORT:-22000}"
RLVE_RAY_RUNTIME_ENV_AGENT_BASE_PORT="${RLVE_RAY_RUNTIME_ENV_AGENT_BASE_PORT:-23000}"
RLVE_RAY_WORKER_MIN_BASE_PORT="${RLVE_RAY_WORKER_MIN_BASE_PORT:-30000}"
# Ray may prestart one worker per CPU; keep the worker range wider than that
# so job drivers can register instead of hanging during submission.
RLVE_RAY_WORKER_PORT_STRIDE="${RLVE_RAY_WORKER_PORT_STRIDE:-200}"

if ! [[ "${RLVE_RAY_PORT_SPAN}" =~ ^[0-9]+$ ]] || [ "${RLVE_RAY_PORT_SPAN}" -lt 1 ]; then
    echo "ERROR: RLVE_RAY_PORT_SPAN must be a positive integer, got '${RLVE_RAY_PORT_SPAN}'."
    exit 1
fi
if ! [[ "${RLVE_RAY_WORKER_PORT_STRIDE}" =~ ^[0-9]+$ ]] || [ "${RLVE_RAY_WORKER_PORT_STRIDE}" -lt 2 ]; then
    echo "ERROR: RLVE_RAY_WORKER_PORT_STRIDE must be an integer >=2, got '${RLVE_RAY_WORKER_PORT_STRIDE}'."
    exit 1
fi

PORT_SLOT=0
if [ "${RLVE_AUTO_RAY_PORT_BY_GPU}" = "1" ]; then
    PORT_HASH=$(echo -n "${RLVE_RAY_CLUSTER_KEY}" | cksum | awk '{print $1}')
    PORT_SLOT=$((PORT_HASH % RLVE_RAY_PORT_SPAN))
fi

USER_RAY_DASHBOARD_URL="${RAY_DASHBOARD_URL:-}"
if [ -n "${RLVE_RAY_DASHBOARD_URL:-}" ]; then
    RAY_DASHBOARD_URL="${RLVE_RAY_DASHBOARD_URL}"
elif [ -n "${USER_RAY_DASHBOARD_URL}" ]; then
    RAY_DASHBOARD_URL="${USER_RAY_DASHBOARD_URL}"
fi

if [ -n "${RAY_DASHBOARD_URL:-}" ]; then
    RAY_DASHBOARD_PORT="${RAY_DASHBOARD_URL##*:}"
else
    RAY_DASHBOARD_PORT="${RLVE_RAY_DASHBOARD_PORT:-$((RLVE_RAY_DASHBOARD_BASE_PORT + PORT_SLOT))}"
    RAY_DASHBOARD_URL="http://127.0.0.1:${RAY_DASHBOARD_PORT}"
fi

RAY_GCS_PORT="${RLVE_RAY_GCS_PORT:-$((RLVE_RAY_GCS_BASE_PORT + PORT_SLOT))}"
RAY_CLIENT_SERVER_PORT="${RLVE_RAY_CLIENT_SERVER_PORT:-$((RLVE_RAY_CLIENT_BASE_PORT + PORT_SLOT))}"
RAY_METRICS_EXPORT_PORT="${RLVE_RAY_METRICS_EXPORT_PORT:-$((RLVE_RAY_METRICS_BASE_PORT + PORT_SLOT))}"
RAY_NODE_MANAGER_PORT="${RLVE_RAY_NODE_MANAGER_PORT:-$((RLVE_RAY_NODE_MANAGER_BASE_PORT + PORT_SLOT))}"
RAY_OBJECT_MANAGER_PORT="${RLVE_RAY_OBJECT_MANAGER_PORT:-$((RLVE_RAY_OBJECT_MANAGER_BASE_PORT + PORT_SLOT))}"
RAY_DASHBOARD_AGENT_LISTEN_PORT="${RLVE_RAY_DASHBOARD_AGENT_LISTEN_PORT:-$((RLVE_RAY_DASHBOARD_AGENT_LISTEN_BASE_PORT + PORT_SLOT))}"
RAY_DASHBOARD_AGENT_GRPC_PORT="${RLVE_RAY_DASHBOARD_AGENT_GRPC_PORT:-$((RLVE_RAY_DASHBOARD_AGENT_GRPC_BASE_PORT + PORT_SLOT))}"
RAY_RUNTIME_ENV_AGENT_PORT="${RLVE_RAY_RUNTIME_ENV_AGENT_PORT:-$((RLVE_RAY_RUNTIME_ENV_AGENT_BASE_PORT + PORT_SLOT))}"
RAY_MIN_WORKER_PORT="${RLVE_RAY_MIN_WORKER_PORT:-$((RLVE_RAY_WORKER_MIN_BASE_PORT + PORT_SLOT * RLVE_RAY_WORKER_PORT_STRIDE))}"
RAY_MAX_WORKER_PORT="${RLVE_RAY_MAX_WORKER_PORT:-$((RAY_MIN_WORKER_PORT + RLVE_RAY_WORKER_PORT_STRIDE - 1))}"
RAY_TEMP_DIR="${RLVE_RAY_TEMP_DIR:-/tmp/rlve_ray_${RAY_GCS_PORT}}"

validate_port() {
    local name="$1"
    local value="$2"
    if ! [[ "${value}" =~ ^[0-9]+$ ]] || [ "${value}" -lt 1024 ] || [ "${value}" -gt 65535 ]; then
        echo "ERROR: ${name} must be an integer in [1024, 65535], got '${value}'."
        exit 1
    fi
}

validate_port "RAY_DASHBOARD_PORT" "${RAY_DASHBOARD_PORT}"
validate_port "RAY_GCS_PORT" "${RAY_GCS_PORT}"
validate_port "RAY_CLIENT_SERVER_PORT" "${RAY_CLIENT_SERVER_PORT}"
validate_port "RAY_METRICS_EXPORT_PORT" "${RAY_METRICS_EXPORT_PORT}"
validate_port "RAY_NODE_MANAGER_PORT" "${RAY_NODE_MANAGER_PORT}"
validate_port "RAY_OBJECT_MANAGER_PORT" "${RAY_OBJECT_MANAGER_PORT}"
validate_port "RAY_DASHBOARD_AGENT_LISTEN_PORT" "${RAY_DASHBOARD_AGENT_LISTEN_PORT}"
validate_port "RAY_DASHBOARD_AGENT_GRPC_PORT" "${RAY_DASHBOARD_AGENT_GRPC_PORT}"
validate_port "RAY_RUNTIME_ENV_AGENT_PORT" "${RAY_RUNTIME_ENV_AGENT_PORT}"
validate_port "RAY_MIN_WORKER_PORT" "${RAY_MIN_WORKER_PORT}"
validate_port "RAY_MAX_WORKER_PORT" "${RAY_MAX_WORKER_PORT}"
if [ "${RAY_MAX_WORKER_PORT}" -le "${RAY_MIN_WORKER_PORT}" ]; then
    echo "ERROR: RAY_MAX_WORKER_PORT (${RAY_MAX_WORKER_PORT}) must be greater than RAY_MIN_WORKER_PORT (${RAY_MIN_WORKER_PORT})."
    exit 1
fi

echo "RAY_CLUSTER_KEY=${RLVE_RAY_CLUSTER_KEY} (PORT_SLOT=${PORT_SLOT}, GPU_SIGNATURE=${VISIBLE_GPU_SIGNATURE})"
echo "RAY_DASHBOARD_URL=${RAY_DASHBOARD_URL}"
echo "RAY_PORTS: gcs=${RAY_GCS_PORT}, dashboard=${RAY_DASHBOARD_PORT}, client=${RAY_CLIENT_SERVER_PORT}, metrics=${RAY_METRICS_EXPORT_PORT}, node_manager=${RAY_NODE_MANAGER_PORT}, object_manager=${RAY_OBJECT_MANAGER_PORT}, dashboard_agent_listen=${RAY_DASHBOARD_AGENT_LISTEN_PORT}, dashboard_agent_grpc=${RAY_DASHBOARD_AGENT_GRPC_PORT}, runtime_env_agent=${RAY_RUNTIME_ENV_AGENT_PORT}, worker_range=${RAY_MIN_WORKER_PORT}-${RAY_MAX_WORKER_PORT}"
echo "RAY_TEMP_DIR=${RAY_TEMP_DIR}"

# Persist lightweight runtime metadata per GCS port to prevent incompatible reuse.
RAY_RUNTIME_META_FILE="/tmp/rlve_ray_runtime_${RAY_GCS_PORT}.meta"
RAY_RUNTIME_META_VERSION=3
RLVE_ALLOW_REUSE_WITHOUT_META="${RLVE_ALLOW_REUSE_WITHOUT_META:-0}"
if ray job list --address="${RAY_DASHBOARD_URL}" >/dev/null 2>&1; then
    echo "Reusing existing Ray runtime at ${RAY_DASHBOARD_URL}"
    if [ -f "${RAY_RUNTIME_META_FILE}" ]; then
        PREV_RAY_RUNTIME_META_VERSION=$(awk -F= '/^RAY_RUNTIME_META_VERSION=/{print $2}' "${RAY_RUNTIME_META_FILE}")
        PREV_VISIBLE_GPU_SIGNATURE=$(awk -F= '/^VISIBLE_GPU_SIGNATURE=/{print $2}' "${RAY_RUNTIME_META_FILE}")
        PREV_RAY_GCS_PORT=$(awk -F= '/^RAY_GCS_PORT=/{print $2}' "${RAY_RUNTIME_META_FILE}")
        if [ "${PREV_RAY_RUNTIME_META_VERSION:-1}" != "${RAY_RUNTIME_META_VERSION}" ]; then
            echo "ERROR: Ray runtime metadata version mismatch on ${RAY_DASHBOARD_URL}."
            echo "ERROR: existing_meta_version=${PREV_RAY_RUNTIME_META_VERSION:-1}, required_meta_version=${RAY_RUNTIME_META_VERSION}."
            echo "ERROR: This runtime was started without isolated Ray agent ports. Use a fresh Ray runtime for this GPU set."
            exit 1
        fi
        if [ -n "${PREV_RAY_GCS_PORT}" ] && [ "${PREV_RAY_GCS_PORT}" != "${RAY_GCS_PORT}" ]; then
            echo "ERROR: Ray runtime GCS port mismatch on ${RAY_DASHBOARD_URL}."
            echo "ERROR: existing_gcs_port=${PREV_RAY_GCS_PORT}, current_gcs_port=${RAY_GCS_PORT}."
            echo "ERROR: Set RLVE_RAY_DASHBOARD_URL / RLVE_RAY_GCS_PORT consistently and retry."
            exit 1
        fi
        if [ -n "${PREV_VISIBLE_GPU_SIGNATURE}" ] && [ "${PREV_VISIBLE_GPU_SIGNATURE}" != "${VISIBLE_GPU_SIGNATURE}" ]; then
            echo "ERROR: Ray runtime GPU signature mismatch on ${RAY_DASHBOARD_URL}."
            echo "ERROR: existing=${PREV_VISIBLE_GPU_SIGNATURE}, current=${VISIBLE_GPU_SIGNATURE}."
            echo "ERROR: Set RLVE_RAY_DASHBOARD_URL to another port (or stop old Ray runtime) and retry."
            exit 1
        fi
    elif [ "${RLVE_ALLOW_REUSE_WITHOUT_META}" != "1" ]; then
        echo "ERROR: Found Ray runtime at ${RAY_DASHBOARD_URL}, but metadata ${RAY_RUNTIME_META_FILE} is missing."
        echo "ERROR: To avoid unsafe reuse, set RLVE_RAY_DASHBOARD_URL to a fresh port and retry."
        echo "ERROR: Or set RLVE_ALLOW_REUSE_WITHOUT_META=1 to bypass this check."
        exit 1
    else
        echo "WARN: Reusing runtime without metadata check (RLVE_ALLOW_REUSE_WITHOUT_META=1)."
    fi
else
    echo "No active Ray runtime detected; starting a new head node on port ${RAY_DASHBOARD_PORT}."
    ray start --head \
        --node-ip-address "${MASTER_ADDR}" \
        --num-gpus "${NUM_GPUS}" \
        --port "${RAY_GCS_PORT}" \
        --dashboard-host=0.0.0.0 \
        --dashboard-port "${RAY_DASHBOARD_PORT}" \
        --ray-client-server-port "${RAY_CLIENT_SERVER_PORT}" \
        --metrics-export-port "${RAY_METRICS_EXPORT_PORT}" \
        --node-manager-port "${RAY_NODE_MANAGER_PORT}" \
        --object-manager-port "${RAY_OBJECT_MANAGER_PORT}" \
        --dashboard-agent-listen-port "${RAY_DASHBOARD_AGENT_LISTEN_PORT}" \
        --dashboard-agent-grpc-port "${RAY_DASHBOARD_AGENT_GRPC_PORT}" \
        --runtime-env-agent-port "${RAY_RUNTIME_ENV_AGENT_PORT}" \
        --min-worker-port "${RAY_MIN_WORKER_PORT}" \
        --max-worker-port "${RAY_MAX_WORKER_PORT}" \
        --temp-dir "${RAY_TEMP_DIR}" \
        --disable-usage-stats
    cat > "${RAY_RUNTIME_META_FILE}" <<EOF
RAY_RUNTIME_META_VERSION=${RAY_RUNTIME_META_VERSION}
VISIBLE_GPU_SIGNATURE=${VISIBLE_GPU_SIGNATURE}
NUM_GPUS=${NUM_GPUS}
RAY_GCS_PORT=${RAY_GCS_PORT}
RAY_DASHBOARD_URL=${RAY_DASHBOARD_URL}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT}
RAY_CLIENT_SERVER_PORT=${RAY_CLIENT_SERVER_PORT}
RAY_METRICS_EXPORT_PORT=${RAY_METRICS_EXPORT_PORT}
RAY_NODE_MANAGER_PORT=${RAY_NODE_MANAGER_PORT}
RAY_OBJECT_MANAGER_PORT=${RAY_OBJECT_MANAGER_PORT}
RAY_DASHBOARD_AGENT_LISTEN_PORT=${RAY_DASHBOARD_AGENT_LISTEN_PORT}
RAY_DASHBOARD_AGENT_GRPC_PORT=${RAY_DASHBOARD_AGENT_GRPC_PORT}
RAY_RUNTIME_ENV_AGENT_PORT=${RAY_RUNTIME_ENV_AGENT_PORT}
RAY_MIN_WORKER_PORT=${RAY_MIN_WORKER_PORT}
RAY_MAX_WORKER_PORT=${RAY_MAX_WORKER_PORT}
RAY_TEMP_DIR=${RAY_TEMP_DIR}
EOF
fi

# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"max_split_size_mb:1024\"
  }
}"

RLVE_RAY_READY_RETRIES="${RLVE_RAY_READY_RETRIES:-60}"
RLVE_RAY_READY_SLEEP_SECONDS="${RLVE_RAY_READY_SLEEP_SECONDS:-5}"
RAY_READY_COUNT=0
until timeout 10 ray job list --address="${RAY_DASHBOARD_URL}" >/dev/null 2>&1; do
    if [ "${RAY_READY_COUNT}" -ge "${RLVE_RAY_READY_RETRIES}" ]; then
        echo "ERROR: Ray job server did not become ready at ${RAY_DASHBOARD_URL}."
        exit 1
    fi
    echo "Waiting for Ray job server at ${RAY_DASHBOARD_URL}... ($((RAY_READY_COUNT+1))/${RLVE_RAY_READY_RETRIES})"
    sleep "${RLVE_RAY_READY_SLEEP_SECONDS}"
    RAY_READY_COUNT=$((RAY_READY_COUNT + 1))
done

MAX_SUBMIT_RETRIES="${RLVE_RAY_SUBMIT_RETRIES:-20}"
RLVE_RAY_SUBMIT_RETRY_SLEEP_SECONDS="${RLVE_RAY_SUBMIT_RETRY_SLEEP_SECONDS:-5}"
RLVE_RAY_SUBMIT_TIMEOUT_SECONDS="${RLVE_RAY_SUBMIT_TIMEOUT_SECONDS:-60}"
SUBMIT_RETRY_COUNT=0
while true; do
    SUBMIT_EXIT_CODE=0
    SUBMIT_OUTPUT=$(timeout "${RLVE_RAY_SUBMIT_TIMEOUT_SECONDS}" ray job submit --no-wait --address="${RAY_DASHBOARD_URL}" \
       --runtime-env-json="${RUNTIME_ENV_JSON}" \
       -- python3 train.py \
       --actor-num-nodes 1 \
       --actor-num-gpus-per-node "${NUM_GPUS}" \
       --colocate \
       ${MODEL_ARGS[@]} \
       ${CKPT_ARGS[@]} \
       ${ROLLOUT_ARGS[@]} \
       ${OPTIMIZER_ARGS[@]} \
       ${GRPO_ARGS[@]} \
       ${DISTRIBUTED_ARGS[@]} \
       ${WANDB_ARGS[@]} \
       ${PERF_ARGS[@]} \
       ${EVAL_ARGS[@]} \
       ${SGLANG_ARGS[@]} \
       ${MISC_ARGS[@]} \
       "${EXTRA_ARGS[@]}" 2>&1) || SUBMIT_EXIT_CODE=$?

    RAY_JOB_ID=$(echo "${SUBMIT_OUTPUT}" | python3 -c 'import re, sys
text = sys.stdin.read()
match = re.search(r"(raysubmit_[A-Za-z0-9_-]+)", text)
if not match:
    match = re.search(r"Job '\''([^'\'']+)'\'' submitted successfully", text)
if match:
    print(match.group(1))
')

    if [ -n "${RAY_JOB_ID}" ]; then
        echo "${SUBMIT_OUTPUT}" | tee -a "${RAY_JOB_LOG}"
        break
    fi

    if [ "${SUBMIT_EXIT_CODE}" = "124" ] || echo "${SUBMIT_OUTPUT}" | grep -E -i -q "status code 500|status code 502|status code 503|status code 504|Connection refused|No available agent|timed out"; then
        if [ "$SUBMIT_RETRY_COUNT" -ge "$MAX_SUBMIT_RETRIES" ]; then
            echo "ERROR: Failed to submit Ray job after ${MAX_SUBMIT_RETRIES} retries. Last output:"
            echo "${SUBMIT_OUTPUT}" | tee -a "${RAY_JOB_LOG}"
            exit 1
        fi
        echo "Transient error during job submission (agent not ready or request timed out). Retrying in ${RLVE_RAY_SUBMIT_RETRY_SLEEP_SECONDS} seconds... ($((SUBMIT_RETRY_COUNT+1))/${MAX_SUBMIT_RETRIES})"
        sleep "${RLVE_RAY_SUBMIT_RETRY_SLEEP_SECONDS}"
        SUBMIT_RETRY_COUNT=$((SUBMIT_RETRY_COUNT + 1))
    else
        echo "ERROR: Job submission failed with an unexpected error:"
        echo "${SUBMIT_OUTPUT}" | tee -a "${RAY_JOB_LOG}"
        exit 1
    fi
done

if [ -z "${RAY_JOB_ID}" ]; then
    echo "ERROR: Unable to parse Ray job id from submission output."
    exit 1
fi
echo "RAY_JOB_ID=${RAY_JOB_ID}"

# Stream Ray job logs to a dedicated file while keeping terminal output.
ray job logs -f "${RAY_JOB_ID}" --address="${RAY_DASHBOARD_URL}" 2>&1 | tee -a "${RAY_JOB_LOG}"
