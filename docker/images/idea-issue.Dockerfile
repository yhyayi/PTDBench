# PTDBench environment `idea-issue`.
# ptdbench-image: ptdbench/idea-issue:py312-20261023-deps1
#
# Static IDEA diagnosis, parquet inspection and the complete package self-tests.
# Optional GPU reproduction uses the separate idea-historical image, not host conda.
# This image intentionally does not claim to reproduce the GPU training stack.
# Tasks: reward/idea_awm_reward_diagnosis.
#
# Like verl-moe-token, this Dockerfile is a recipe the harness never builds:
# `ptdbench build-images` only pulls the tag above. Build it manually from the
# docker/ context (`docker build -f docker/images/idea-issue.Dockerfile docker/`)
# and push/tag as the header declares. No activate hook is baked -- the harness
# injects it at run time -- and fetch tooling ships per task in environment/.
# Official Python linux/amd64 manifest, fetched through a registry mirror.
FROM docker.m.daocloud.io/library/python:3.12-slim-bookworm@sha256:57d96f71de145a98e7ceca18ee4ade77bba4103a33a72dfaea026b23625c6bf0

# IDEA-specific dependencies are embedded: no task checkout, host environment,
# or separate requirements file is needed in the Docker build context.
RUN python3 - <<'PY'
from pathlib import Path

requirements = r"""# Static diagnosis/self-test dependencies; no torch, vLLM, CUDA or conda.
annotated-types==0.8.0
antlr4-python3-runtime==4.9.3
certifi==2026.7.22
charset-normalizer==3.5.1
click==8.4.2
idna==3.19
importlib_metadata==9.0.0
omegaconf==2.3.1
opentelemetry-api==1.44.0
packaging==25.0
platformdirs==4.11.3
protobuf==7.36.0
pyarrow==25.0.1
pydantic==2.13.4
pydantic_core==2.46.4
PyYAML==6.0.3
requests==2.34.2
sentry-sdk==2.68.0
typing-inspection==0.4.4
typing_extensions==4.16.0
urllib3==2.7.0
wandb==0.28.2
zipp==4.1.0
"""
path = Path('/opt/idea-diagnosis/requirements.txt')
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(requirements)
PY
ENV WANDB_MODE=offline \
    WANDB_SILENT=true \
    PYTHONDONTWRITEBYTECODE=1
RUN python3 -m pip install --no-cache-dir --no-deps -r /opt/idea-diagnosis/requirements.txt && \
    python3 -m pip check && \
    python3 -c "import omegaconf, pyarrow.parquet; from wandb.proto import wandb_internal_pb2; from wandb.sdk.internal.datastore import DataStore"

# The harness bind-mounts a per-run copy of the task's workspace here.
WORKDIR /workspace
