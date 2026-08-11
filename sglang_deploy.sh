docker run --gpus '"device=0,1"' \
  --ipc=host \
  --shm-size 32g \
  -p 30004:30004 \
  -v /home/test/test1706/yuanjiarui/projects/harnessRL/checkpoints/Qwen3.5-9B:/model \
  lmsysorg/sglang:latest \
  python3 -m sglang.launch_server \
    --model-path /model \
    --host 0.0.0.0 \
    --port 30004 \
    --enable-dp-attention \
    --tp-size 2 \
    --dp-size 2 \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder