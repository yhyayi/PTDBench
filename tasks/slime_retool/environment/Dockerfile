# ReTool recipe diagnosis runs on top of the public slime image so that the
# workspace code and the verifier's Python (PyYAML) are available. This is a
# code/configuration assessment: it does NOT rerun training and therefore does
# NOT download the Qwen3-4B checkpoint, the DAPO-Math-17k / ReTool-SFT data, or
# execute the code-interpreter sandbox. Those assets are not committed to the
# repository and are not needed to review the recipe.
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
