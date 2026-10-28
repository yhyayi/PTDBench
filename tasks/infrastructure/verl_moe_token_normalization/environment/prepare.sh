#!/usr/bin/env bash
# Recorded assets are immutable. Every downloaded file is checked before conversion.
set -euo pipefail
: "${PTDBENCH_WORKSPACE:?}"
export W="$PTDBENCH_WORKSPACE"
mkdir -p "$W/metadata" "$W/data/gsm8k" "$W/.tmp" "$W/.home"
export HOME="$W/.home" TMPDIR="$W/.tmp"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export PYTHONPATH="$W/src/$(cat "$W/variant.txt"):$W/scripts"
python3 "$PTDBENCH_ENVIRONMENT/ptdbench_fetch.py" \
    --repo Qwen/Qwen1.5-MoE-A2.7B-Chat \
    --type model \
    --dest "$W/models/Qwen1.5-MoE-A2.7B-Chat"

# Record the hash-verified full model's parameter count so runtime scripts can
# size the shared-GPU capacity wait without re-reading every shard header.
python3 - "$W/models/Qwen1.5-MoE-A2.7B-Chat" "$W/metadata/REAL_READY.json" <<'PYEOF'
import json
import math
import pathlib
import sys

model_dir, ready_path = sys.argv[1], sys.argv[2]
total = 0
for shard in sorted(pathlib.Path(model_dir).glob("*.safetensors")):
    with open(shard, "rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(header_len))
    total += sum(math.prod(t["shape"]) for k, t in header.items() if k != "__metadata__")
assert total == 14315784192, f"unexpected parameter count: {total}"
with open(ready_path, "w") as f:
    json.dump({"model_id": "Qwen/Qwen1.5-MoE-A2.7B-Chat", "parameters": total}, f)
    f.write("\n")
PYEOF

python3 "$PTDBENCH_ENVIRONMENT/ptdbench_fetch.py" \
    --repo openai/gsm8k \
    --type dataset \
    --dest "$W/data/raw/gsm8k"

python3 "$W/scripts/prepare_gsm8k.py"
