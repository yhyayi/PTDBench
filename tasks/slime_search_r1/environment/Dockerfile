# Search-R1 recipe diagnosis runs on top of the public slime image so that the
# workspace code and the verifier's Python (PyYAML) are available. This is a
# code/configuration assessment: it does NOT rerun training and therefore does
# NOT download the Qwen2.5-3B checkpoint, the nq_hotpotqa_train Parquet data, or
# the wiki-18 FAISS retrieval index. Those assets are not committed to the
# repository and are not needed to review the recipe.
FROM slimerl/slime:latest

LABEL org.ptdbench.task="slime_search_r1" \
      org.ptdbench.framework="slime" \
      org.ptdbench.recipe="examples/search-r1"

# Guard: ensure the verifier's only third-party import (PyYAML) is present even
# if a future base image drops it. The slime image already ships Python 3.
RUN python3 -c "import yaml" 2>/dev/null || pip install --no-cache-dir pyyaml tensordict chardet faiss-cpu

WORKDIR /workspace
CMD ["tail", "-f", "/dev/null"]
