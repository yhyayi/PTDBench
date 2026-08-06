PYTHONDONTWRITEBYTECODE=1 python3 -m ptdbench evaluate example/swe_slugfix \
  --backend local \
  --agent-backend local \
  --agent codex \
  --model gpt-5.5 \
  --base-url "https://yeysai.com/v1" \
  --api-key "sk-FZzgnPyqBCjlLJbwrU98cMiuxiGcMfhxgpKuLJkhuCtye49T" \
  --work-dir outputs/ptdbench-swe-debug \
  --keep-artifacts \
  --json