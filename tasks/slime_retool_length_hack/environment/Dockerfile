# ReTool recipe diagnosis and repair-and-rerun on the public slime image.
# Issue grading is log/code review. Solution reruns still need host-side
# Qwen3-4B-SFT weights and DAPO-Math / AIME jsonl already present under
# /workspace. The image does not download those assets.
FROM slimerl/slime:latest

LABEL org.ptdbench.task="slime_retool" \
      org.ptdbench.framework="slime" \
      org.ptdbench.recipe="examples/retool"

# Guard: ensure the verifier's third-party import (PyYAML) plus the recipe's
# declared deps (jinja2 for the tool prompt template, psutil for the sandbox)
# are present even if a future base image drops them. The slime image already
# ships Python 3.
RUN python3 -c "import yaml" 2>/dev/null || pip install --no-cache-dir pyyaml
RUN python3 -c "import jinja2, psutil" 2>/dev/null || pip install --no-cache-dir jinja2 psutil

WORKDIR /workspace
CMD ["tail", "-f", "/dev/null"]
