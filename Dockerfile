# =============================================================================
# ACE-Step 1.5 — Generic CUDA Dockerfile
# =============================================================================
#
# Builds ACE-Step 1.5 for x86_64 Linux servers with NVIDIA GPUs.
# Uses uv for fast, reproducible dependency installation.
#
# Build:
#   docker build -t acestep .
#
# Run (REST API server — default). ASS drives this in production and supplies the
# real cache paths via env; a standalone smoke run just needs the one /cache mount
# and the port (2766 = 0xACE):
#   docker run --gpus all -it --rm \
#     -p 2766:2766 \
#     -v /srv/ass/cache:/cache \
#     -e ACESTEP_CHECKPOINTS_DIR=/cache/acestep/checkpoints \
#     acestep
#
# Run (Gradio UI instead):
#   docker run --gpus all -it --rm \
#     -p 7860:7860 \
#     -v /srv/ass/cache:/cache \
#     -e ACESTEP_MODE=gradio \
#     acestep
#
# =============================================================================

# ==================== Build arguments ====================
ARG CUDA_VERSION=12.8.1
ARG PYTHON_VERSION=3.11
ARG UV_VERSION=0.7

# ==================== Base image ====================
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# ==================== System packages ====================
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        build-essential \
        git \
        curl \
        wget \
        # Audio libraries
        libsndfile1 \
        libsndfile1-dev \
        ffmpeg \
        # Python build deps
        libffi-dev \
        libssl-dev \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3.11-venv \
    && rm -rf /var/lib/apt/lists/*

# ==================== uv ====================
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# ==================== Install dependencies via uv ====================
# Deps BEFORE source, so a source edit doesn't invalidate the multi-GB torch/CUDA
# layer and re-push it (that was the 30-min-per-commit tax). Copy only the manifests
# — plus the one local path dependency (nano-vllm) uv.lock points at, which must be
# present for the frozen sync. --no-install-project installs the deps but not the
# ace-step package itself; that lands in the fast source layer below.
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY acestep/third_parts/nano-vllm ./acestep/third_parts/nano-vllm
RUN uv sync --frozen --no-dev --python python3.11 --no-install-project

# ==================== Project source ====================
# Just the source now — a tiny layer. Every future code change rebuilds/pushes
# this in seconds instead of dragging the whole dependency set along.
COPY . /app/
RUN uv sync --frozen --no-dev --python python3.11

# ==================== Runtime directories ====================
RUN mkdir -p \
        /app/checkpoints \
        /app/.cache/acestep/tmp \
        /app/gradio_outputs \
        /app/output \
        /app/lokr_output

# ==================== Environment ====================
# Bind to all interfaces for Docker port-mapping
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV ACESTEP_API_HOST=0.0.0.0

# Default startup mode: "api" for the REST server, "gradio" for the web UI
ENV ACESTEP_MODE=api

# Auto-initialize API models according to detected GPU capability. Set
# ACESTEP_NO_INIT=true to defer initialization until the first request.
ENV ACESTEP_NO_INIT=false
ENV ACESTEP_INIT_LLM=auto

# Default models. XL (4B) turbo is the DiT everyone actually runs now — the old
# 2B acestep-v15-turbo is retired. Paired with the 1.7B planner (the 4B LM was
# only ever the xl-sft experiment). pt LM backend, not vllm: pt offloads the
# planner to CPU between steps (vllm greedily pins VRAM AND takes ~60s to init an
# engine that park can't move), which is what lets the card be shared / parked.
# NOTE the var name: the code reads ACESTEP_LM_BACKEND and DEFAULTS TO vllm if it's
# unset — the old ACESTEP_LLM_BACKEND here was a typo that set nothing, so a
# standalone run silently got vllm. ASS overrides it correctly; this fixes the
# image's own default so standalone runs get pt too.
ENV ACESTEP_CONFIG_PATH=acestep-v15-xl-turbo
ENV ACESTEP_LM_MODEL_PATH=acestep-5Hz-lm-1.7B
ENV ACESTEP_LM_BACKEND=pt

# Keep runtime caches below a single mountable application cache directory.
ENV ACESTEP_TMPDIR=/app/.cache/acestep/tmp
ENV TRITON_CACHE_DIR=/app/.cache/acestep/triton
ENV TORCHINDUCTOR_CACHE_DIR=/app/.cache/acestep/torchinductor
# ASS cache convention: caches live under /cache (mounted from the host at
# /srv/ass/cache). These are GENERIC defaults for a standalone run; under ASS each
# service overrides them to its OWN per-service subdir (HF_HOME=/cache/acestep/
# huggingface, TORCH_HOME=/cache/acestep/torch, ACESTEP_CHECKPOINTS_DIR=/cache/
# acestep/checkpoints) so its weights are one deletable folder. The HF token is NOT
# per-service: every service reads one shared file via HF_TOKEN_PATH=/cache/hf-token.
ENV HF_HOME=/cache/huggingface
ENV TORCH_HOME=/cache/torch
ENV HF_TOKEN_PATH=/cache/hf-token

# Disable tokenizers parallelism warnings
ENV TOKENIZERS_PARALLELISM=false

# ==================== Ports ====================
# 7860 = Gradio web UI | 2766 = REST API server (0xACE)
EXPOSE 7860 2766

# ==================== Health check ====================
HEALTHCHECK --interval=60s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -sf http://localhost:${GRADIO_PORT:-7860}/ > /dev/null 2>&1 \
     || curl -sf http://localhost:${ACESTEP_API_PORT:-2766}/health > /dev/null 2>&1 \
     || exit 1

# ==================== Entrypoint ====================
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

ENTRYPOINT ["/app/docker-entrypoint.sh"]
