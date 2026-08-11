PYTHONDONTWRITEBYTECODE=1 python3 -m ptdbench evaluate example/swe_slugfix \
  --backend local \
  --agent-backend local \
  --agent codex \
  --model /model \
  --base-url "http://127.0.0.1:30004/v1" \
  --api-key "EMPTY" \
  --work-dir outputs/ptdbench-swe-debug \
  --keep-artifacts \
  --json
