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
# Run (REST API server — default):
#   docker run --gpus all -it --rm \
#     -p 8001:8001 \
#     -v /srv/acestep/checkpoints:/app/checkpoints \
#     -v /srv/acestep/cache:/app/.cache/acestep \
#     -v /srv/acestep/huggingface:/root/.cache/huggingface \
#     acestep
#
# Run (Gradio UI instead):
#   docker run --gpus all -it --rm \
#     -p 7860:7860 \
#     -v /srv/acestep/checkpoints:/app/checkpoints \
#     -v /srv/acestep/gradio_outputs:/app/gradio_outputs \
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

# ==================== Project source ====================
WORKDIR /app
COPY . /app/

# ==================== Install dependencies via uv ====================
# Use uv sync with the lockfile for reproducible builds.
# --no-dev skips dev dependencies, --frozen uses exact lockfile versions.
RUN uv sync --frozen --no-dev --python python3.11

# ==================== Runtime directories ====================
RUN mkdir -p \
        /app/checkpoints \
        /app/.cache/acestep/tmp \
        /app/gradio_outputs \
        /app/output \
        /app/lokr_output \
        /root/.cache/huggingface

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

# Default models
ENV ACESTEP_CONFIG_PATH=acestep-v15-turbo
ENV ACESTEP_LM_MODEL_PATH=acestep-5Hz-lm-1.7B
ENV ACESTEP_LLM_BACKEND=pt

# Keep runtime caches below a single mountable application cache directory.
ENV ACESTEP_TMPDIR=/app/.cache/acestep/tmp
ENV TRITON_CACHE_DIR=/app/.cache/acestep/triton
ENV TORCHINDUCTOR_CACHE_DIR=/app/.cache/acestep/torchinductor
# ASS shared-cache convention: every backend image caches HF/torch weights under
# /cache, and every service mounts the SAME host dir there (/srv/ass/cache:/cache).
# One shared cache means the HF token is written once ($HF_HOME/token) and weights
# that share a base model are deduplicated across backends. (Was /root/.cache/*.)
ENV HF_HOME=/cache/huggingface
ENV TORCH_HOME=/cache/torch

# Disable tokenizers parallelism warnings
ENV TOKENIZERS_PARALLELISM=false

# ==================== Ports ====================
# 7860 = Gradio web UI | 8001 = REST API server
EXPOSE 7860 8001

# ==================== Health check ====================
HEALTHCHECK --interval=60s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -sf http://localhost:${GRADIO_PORT:-7860}/ > /dev/null 2>&1 \
     || curl -sf http://localhost:${ACESTEP_API_PORT:-8001}/health > /dev/null 2>&1 \
     || exit 1

# ==================== Entrypoint ====================
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

ENTRYPOINT ["/app/docker-entrypoint.sh"]
