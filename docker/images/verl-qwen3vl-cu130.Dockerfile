# PTDBench environment verl-qwen3vl-cu130; not yet built.
# ptdbench-image: ptdbench/verl-qwen3vl-cu130:vllm0.24-torch2.11-cu130
# Recorded numerical stack: job 1332901, Qwen3-VL-8B / GRPO / H800.
FROM verlai/verl:vllm024.dev2@sha256:b867883b0dd011363e69ab2ab344922a28c5bd0409e2a324e3ee70fb27ca7543

USER root
ARG PIP_INDEX_URL=https://mirrors.ustc.edu.cn/pypi/simple
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential libnuma1 numactl ffmpeg \
    && rm -rf /var/lib/apt/lists/*
RUN python3 - <<'PYENV'
from pathlib import Path
import subprocess
Path('/opt/ptdbench-qwen3vl.lock').write_text('absl-py==2.5.1\naccelerate==1.13.0\nagent-detector==2.0.0\naiohappyeyeballs==2.7.1\naiohttp==3.14.5\naiohttp-cors==0.8.1\naiosignal==1.4.0\nannotated-doc==0.0.5\nannotated-types==0.8.0\nanthropic==1.17.0\nantlr4-python3-runtime==4.9.3\nanyio==4.15.1\napache-tvm-ffi==0.1.9\nastor==0.8.1\nattrs==26.1.0\nav==19.0.2\nblake3==1.0.11\ncachetools==7.2.4\ncbor2==6.1.5\ncertifi==2026.10.19\ncffi==2.1.1\ncharset-normalizer==3.5.2\nclick==8.5.0\ncloudpickle==3.1.2\ncodetiming==1.4.0\ncolorful==0.5.8\ncompressed-tensors==0.17.0\ncryptography==50.0.2\ncuda-bindings==13.4.4\ncuda-core==1.2.2\ncuda-pathfinder==1.8.3\ncuda-python==13.4.4\ncuda-tile==1.1.0\ncuda-toolkit==13.0.2\ncwevals==0.0.1b1\ndatasets==4.4.1\ndepyf==0.20.0\ndetect-installer==0.2.1\ndill==0.4.0\ndiskcache==5.6.3\ndistlib==0.4.4\ndistro==1.9.0\ndnspython==2.9.0\ndocstring-parser==0.18.0\neinops==0.8.2\nemail-validator==2.3.0\nfastapi==0.136.3\nfastapi-cli==0.0.32\nfastapi-cloud-cli==0.26.0\nfastar==0.12.0\nfastsafetensors==0.5.0\nfilelock==4.2.4\nflashinfer-cubin==0.6.12\nflashinfer-python==0.6.12\nfrozenlist==1.8.0\nfsspec==2025.10.0\ngoogle-api-core==2.42.0\ngoogle-auth==2.62.0\ngoogleapis-common-protos==1.75.5\ngrpcio==1.84.0\nh11==0.16.0\nhf-xet==1.7.0\nhttpcore==1.0.9\nhttpcore2==2.16.0\nhttptools==0.9.0\nhttpx==0.28.1\nhttpx2==2.16.0\nhuggingface-hub==1.33.0\nhumming-kernels==0.1.6\nhydra-core==1.3.2\nidna==3.20\nijson==3.6.0\nimportlib-metadata==9.0.1\ninteregular==0.3.3\njinja2==3.1.6\njiter==0.17.0\njmespath==1.1.0\njsonschema==4.26.0\njsonschema-specifications==2025.9.1\nlark==1.2.2\nlatex2sympy2-extended==1.11.0\nliger-kernel==0.8.5\nllguidance==1.7.6\nllvmlite==0.47.0\nlm-format-enforcer==0.11.3\nloguru==0.7.3\nmarkdown==3.11.2\nmarkdown-it-py==4.2.0\nmarkupsafe==3.0.4\nmath-verify==0.9.0\nmathruler==0.1.0\nmcp==2.3.0\nmcp-types==2.3.0\nmdurl==0.1.2\nmistral-common==1.12.0\nml-dtypes==0.6.0\nmodel-hosting-container-standards==0.1.16\nmpmath==1.3.0\nmsgpack==1.2.3\nmsgspec==0.22.0\nmultidict==7.1.0\nmultiprocess==0.70.18\nnetworkx==3.7\nninja==1.13.2\nnumba==0.65.0\nnumpy==2.2.6\nnvidia-cublas==13.1.0.3\nnvidia-cuda-cccl==13.0.85\nnvidia-cuda-crt==13.0.88\nnvidia-cuda-cupti==13.0.85\nnvidia-cuda-nvcc==13.0.88\nnvidia-cuda-nvrtc==13.0.88\nnvidia-cuda-runtime==13.0.96\nnvidia-cudnn-cu13==9.19.0.56\nnvidia-cudnn-frontend==1.28.0\nnvidia-cufft==12.0.0.61\nnvidia-cufile==1.15.1.6\nnvidia-curand==10.4.0.35\nnvidia-cusolver==12.0.4.66\nnvidia-cusparse==12.6.3.3\nnvidia-cusparselt-cu13==0.8.0\nnvidia-cutlass-dsl==4.5.2\nnvidia-cutlass-dsl-libs-base==4.5.2\nnvidia-cutlass-dsl-libs-cu13==4.5.2\nnvidia-ml-py==13.615.71\nnvidia-nccl-cu13==2.28.9\nnvidia-nvjitlink==13.0.88\nnvidia-nvshmem-cu13==3.4.5\nnvidia-nvtx==13.0.85\nnvidia-nvvm==13.0.88\nomegaconf==2.3.1\nopenai==3.35.1\nopenai-harmony==0.0.8\nopencensus==0.11.4\nopencensus-context==0.1.3\nopencv-python-headless==5.0.0.93\nopentelemetry-api==1.45.1\nopentelemetry-exporter-http-transport==0.66b1\nopentelemetry-exporter-otlp==1.45.1\nopentelemetry-exporter-otlp-common==0.66b1\nopentelemetry-exporter-otlp-proto-common==1.45.1\nopentelemetry-exporter-otlp-proto-grpc==1.45.1\nopentelemetry-exporter-otlp-proto-http==1.45.1\nopentelemetry-exporter-prometheus==0.66b1\nopentelemetry-proto==1.45.1\nopentelemetry-sdk==1.45.1\nopentelemetry-semantic-conventions==0.66b1\nopentelemetry-semantic-conventions-ai==0.5.1\norjson==3.13.0\noutlines-core==0.2.14\npackaging==25.0\npandas==3.0.6\npartial-json-parser==0.2.1.1.post7\npeft==0.18.1\npillow==13.0.0\nplatformdirs==4.13.5\nprometheus-client==0.26.0\nprometheus-fastapi-instrumentator==8.1.0\npropcache==0.5.4\nproto-plus==1.29.0\nprotobuf==7.36.2\npsutil==7.2.2\npy-cpuinfo==9.0.0\npy-spy==0.4.2\npyarrow==26.0.0\npyasn1==0.6.4\npyasn1-modules==0.4.2\npybase64==1.5.1\npybind11==3.1.0\npycountry==26.2.16\npycparser==3.11\npydantic==2.14.1\npydantic-core==2.50.1\npydantic-extra-types==2.11.1\npydantic-settings==2.16.1\npyelftools==0.33\npygments==2.21.0\npyjwt==2.15.1\npylatexenc==2.11\npython-dateutil==2.9.0.post0\npython-discovery==1.6.4\npython-dotenv==1.2.4\npython-json-logger==4.2.0\npython-multipart==0.0.32\npyvers==0.1.0\npyyaml==6.0.3\npyzmq==27.2.0\nquack-kernels==0.5.0\nqwen-vl-utils==0.0.14\nray==2.55.1\nreferencing==0.37.0\nregex==2026.9.29\nrequests==2.34.2\nrich==15.0.0\nrich-toolkit==0.20.6\nrignore==0.8.1\nrpds-py==2026.9.1\nsafetensors==0.8.0\nsentencepiece==0.2.2\nsentry-sdk==2.72.0\nsetproctitle==1.3.8\nsetuptools==80.10.2\nshellingham==1.5.4\nsix==1.17.0\nsmart-open==8.0.4\nsniffio==1.3.1\nsse-starlette==3.5.0\nstarlette==1.9.0\nsupervisor==4.3.0\nsympy==1.14.0\ntabulate==0.10.0\ntensorboard==2.21.0\ntensorboard-data-server==0.7.2\ntensordict==0.10.0\ntiktoken==0.14.0\ntilelang==0.1.9\ntokenizers==0.22.2\ntokenspeed-mla==0.1.2\ntokenspeed-triton==3.9.10.post20261008\ntorch==2.11.0\ntorch-c-dlpack-ext==0.1.5\ntorchaudio==2.11.0\ntorchdata==0.11.0\ntorchvision==0.26.0\ntqdm==4.70.1\ntransferqueue==0.1.8\ntransformers==5.5.4\ntriton==3.6.0\ntruststore==0.10.5\ntyper==0.27.3\ntyping-extensions==4.16.0\ntyping-inspection==0.4.4\nurllib3==2.8.0\nuv==0.9.18\nuvicorn==0.54.0\nuvloop==0.23.0\nvirtualenv==21.14.5\nvllm==0.24.0\nwandb==0.31.0\nwatchfiles==1.3.0\nwebsockets==17.3\nwerkzeug==3.1.9\nwheel==0.48.0\nwrapt==2.5.1\nxgrammar==0.2.3\nxxhash==4.0.2\nyarl==1.25.1\nz3-solver==4.15.4.0\nzipp==4.1.1\n')
Path('/opt/ptdbench-qwen3vl-requirements').write_text("# Core numerical stack. Keep in sync with the official v0.9.0 Docker recipe.\nvllm==0.24.0\ntorch==2.11.0\ntorchvision==0.26.0\ntorchaudio==2.11.0\ntransformers==5.5.4\nnumpy==2.2.6\ntensordict==0.10.0\ndatasets==4.4.1\nhydra-core==1.3.2\nray[default]==2.55.1\naccelerate==1.13.0\npeft==0.18.1\nqwen-vl-utils==0.0.14\ntorchdata==0.11.0\ncodetiming\ndill\npandas\npyarrow>=19\npybind11\npylatexenc\npackaging\nwandb\ntensorboard\nmathruler\nmath-verify\nliger-kernel\nuv==0.9.18\n# An isolated job environment on the selected vllm017 container.\n# Retain the task's original numerical stack; do not inherit system packages.\n\ncuda-toolkit[nvcc,cccl,crt,nvvm]==13.0.2\nTransferQueue==0.1.8\nsetuptools>=77,<81\nwheel\nninja\npsutil\n\n")
subprocess.run(['python3','-m','pip','install','--no-cache-dir','-r','/opt/ptdbench-qwen3vl-requirements','-c','/opt/ptdbench-qwen3vl.lock'],check=True)
subprocess.run(['python3','-m','pip','check'],check=True)
import torch, flash_attn
from importlib.metadata import version
for package, expected in {'torch':'2.11.0','vllm':'0.24.0','transformers':'5.5.4','accelerate':'1.13.0','tensordict':'0.10.0','flash-attn':'2.8.3'}.items():
    assert version(package).split('+')[0] == expected, (package,version(package))
from accelerate import init_empty_weights
p=torch.nn.Parameter(torch.ones(1));p._is_hf_initialized=True
with init_empty_weights():
    m=torch.nn.Module();m.register_parameter('weight',p)
assert m.weight.is_meta and m.weight._is_hf_initialized
PYENV

# ---- PTDBench tooling: keep identical in every docker/images/*.Dockerfile ----
# Build context is docker/ (see docker/.dockerignore for what it may COPY).

RUN mkdir -p /ptdbench/prepared /ptdbench/models /ptdbench/datasets

# Every command the harness runs goes through `sh -lc`, and a login shell's
# /etc/profile may reset PATH. Record the image's own PATH so the hook can put
# back whatever it drops (/usr/local/cuda/bin, /root/.cargo/bin, ...).
RUN printf '%s\n' "$PATH" > /etc/ptdbench-image-path
COPY ptdbench-activate.sh /etc/profile.d/ptdbench-activate.sh
RUN set -eux; \
    sed -i 's/\r$//' /etc/profile.d/ptdbench-activate.sh; \
    chmod 0644 /etc/profile.d/ptdbench-activate.sh; \
    sh -n /etc/profile.d/ptdbench-activate.sh

ENV CUDA_DEVICE_ORDER=PCI_BUS_ID \
    TOKENIZERS_PARALLELISM=false

# The prepare-stage fetch/dataset tooling (ptdbench_fetch.py, ptdbench_dataset.py,
# ptdbench_datasets/) is deliberately NOT baked here. docker/install-ptdbench-tooling.sh
# installs it into /usr/local at the start of every prepare run, from the repo's
# current copy (mounted RO locally, uploaded on Cybertron). That keeps a fix to
# those files reaching running jobs without an image rebuild. Only the activate
# hook above stays baked -- it must exist in the agent/verifier containers too,
# which never run prepare-time provisioning.

# The harness bind-mounts a per-run copy of the task's workspace here.
WORKDIR /workspace
