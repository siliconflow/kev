# kev serving image for SF GPU Functions (4090 g1/g2, kev-{0.8b,4b,9b} routes).
#
# Base choice: kev is a pure PyTorch/peft stack (kev.serve: FastAPI + uvicorn +
# torch + peft + transformers). No vLLM anywhere in the serving path, so this
# does NOT reuse the vllm-openai image — a slim python base with user-space CUDA
# (pip torch wheel bundles its own CUDA runtime libs; host driver is injected by
# the SF GPU Function runtime) keeps the image ~3GB instead of ~25GB.
FROM python:3.12-slim

# gcc/g++ for any source builds in the dep chain (some wheels need a local
# toolchain fallback). git is not needed: weights come from HF Hub / hf-mirror
# at runtime; the source is baked in below.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends gcc g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# 1) Deps layer (cached; changes rarely).
# torch first from the cu124 index (4090 = sm_89; pyproject pins torch>=2.6,<2.9
# and cu124 wheels exist across that range), then the rest from PyPI.
# 2026-09 upstream merge: Qwen3.5 generation (Kev-0.8B/4B/9B) — transformers 5
# (hybrid Gated DeltaNet backbones), peft 0.21, flash-linear-attention for the
# DeltaNet layers on CUDA (README: "install flash-linear-attention for the
# Qwen3.5 models"; without it the recurrent path falls back to a slow one).
COPY pyproject.toml ./
# causal-conv1d: optimized Triton kernel for the Qwen3.5 Gated DeltaNet
# causal-conv path. transformers 5.17 hub_kernels needs the causal_conv1d pip
# distribution present to skip its reference fallback (deploy 2026-09-21 log
# warning). PyPI ships sdist only (needs nvcc); use the prebuilt wheel from
# upstream GitHub Releases instead, matched to the resolved deps:
#   torch 2.6.0+cu124  ->  cu12torch2.6, cxx11abiTRUE, cp312 linux_x86_64.
#   ABI NOTE: cu118/cu124 Linux torch wheels switched to manylinux 2.28 +
#   CXX11_ABI=1 the week of 2024-12-16 (pytorch dev-discuss "PyTorch Linux
#   Wheels switching ...") — BEFORE 2.6.0 GA. So torch 2.6.0+cu124 is a
#   TRUE build; the FALSE wheel (bc0efee/f94c3a7, deployed 09-21) failed to
#   import in-pod and hub_kernels swallowed the ImportError -> the
#   "falling back" warning persisted (deploy log line 216).
# (If the URL ever 404s, pip falls back to the sdist; the gcc/g++ from the
# apt layer then fail fast at build time, not silently at runtime.)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 \
      "torch>=2.6,<2.9" \
    && pip install --no-cache-dir \
      "accelerate>=1.15.0" "datasets>=3.0" "numpy>=2.5.3" "peft>=0.21" \
      "pydantic>=2.9" "scikit-learn>=1.9.1" "transformers>=5.17,<6" \
      "fastapi>=0.115" "typesafe-sdk>=0.6.0" "uvicorn>=0.30" \
      "flash-linear-attention" "triton>=3.7.1" \
    && pip install --no-cache-dir \
      "causal-conv1d @ https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0+cu12torch2.6cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"

# 2) Source layer (installed separately so source-only commits rebuild fast).
COPY kev ./kev
RUN pip install --no-cache-dir --no-deps .

# Non-root runtime, same convention as the vLLM agent images.
RUN useradd --create-home --shell /bin/bash appuser
USER appuser

# Weights are pulled from HF Hub at runtime via HF_ENDPOINT=hf-mirror.com;
# that mirror only fronts plain HTTP downloads. The hub's Xet middleware
# would bypass it and hit cas-server-xethub.hf.sc4.ai directly (unreachable
# from CN networks) -> snapshot_download ConnectionError. Disable Xet.
ENV HF_HUB_DISABLE_XET=1

EXPOSE 8000
# Fragmentation is the default failure mode of a 24 GB card serving variable-size batches
# (2026-09-25 production: 6.04 GiB reserved-but-unallocated while graph captures failed for want
# of contiguous VRAM). expandable_segments lets the allocator grow/shrink segments instead of
# stranding slack in fixed blocks. Override at runtime with -e PYTORCH_CUDA_ALLOC_CONF=... if a
# workload prefers the stock allocator.
ENV PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# The SF cloud-function yaml overrides `command` for serve flags
# (see deploy/kev-4b-4090.yaml in os_jev_exp).
CMD ["python", "-m", "kev.serve", "--run", "jaredpalmer/kev-4b", "--port", "8000"]

# Container-level liveness: /healthz (process + model thread), 3 failures -> unhealthy.
# The deep check is /healthz/ready — point the platform's readiness probe there, not here:
# it runs one real forward pass through the same path traffic takes and is what
# "true reflection of the instance's serving capability" means (a CUDA-OOMed instance
# answers GET endpoints with 200 forever; 2026-09-25: 4,288 /metrics polls green across
# a 5% hard-error window). start-period covers weight loading (ModelScope + HF, ~4 min cold).
HEALTHCHECK --interval=30s --timeout=5s --start-period=420s --retries=3 \
  CMD ["python", "-c", "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/healthz', timeout=4)"]
