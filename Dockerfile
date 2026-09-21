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
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 \
      "torch~=2.8.0" \
    && pip install --no-cache-dir \
      "accelerate>=1.15.0" "datasets>=3.0" "numpy>=2.5.3" "peft>=0.21" \
      "pydantic>=2.9" "scikit-learn>=1.9.1" "transformers>=5.17,<6" \
      "fastapi>=0.115" "typesafe-sdk>=0.6.0" "uvicorn>=0.30" \
      "flash-linear-attention" "triton>=3.7.1" \
    # causal-conv1d: optimized Triton kernel for the Qwen3.5 Gated DeltaNet
    # causal-conv path. PyPI ships sdist only (needs nvcc); use the prebuilt
    # cp312 + cu12 torch2.8 + cxx11abiTRUE wheel from upstream GitHub Releases
    # instead (pip falls back to sdist if the URL ever 404s; gcc/g++ from the
    # apt layer then fail fast at build time, not silently at runtime).
    # transformers 5.17 hub_kernels only needs the causal_conv1d pip distribution
    # present to skip the reference fallback (deploy 2026-09-21 log warning).
    && pip install --no-cache-dir \
      "causal-conv1d @ https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"

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
# The SF cloud-function yaml overrides `command` for serve flags
# (see deploy/kev-4b-4090.yaml in os_jev_exp).
CMD ["python", "-m", "kev.serve", "--run", "jaredpalmer/kev-4b", "--port", "8000"]
