# syntax=docker/dockerfile:1.7

# ============================================================================
# Build arguments
#   CUDA_VERSION    — base image CUDA version. Spark-TTS pins torch==2.5.1,
#                     so the default matches the cu124 wheel (12.4.1).
#   CUDA_FLAVOR     — cuDNN tag segment. 12.4+ uses ``cudnn-runtime``.
#   UBUNTU_VERSION  — base OS.
#   TORCH_VERSION   — upstream Spark-TTS requirement.
#   TORCH_CUDA_TAG  — PyTorch wheel index tag matching CUDA_VERSION.
# ============================================================================
ARG CUDA_VERSION=12.4.1
ARG CUDA_FLAVOR=cudnn-runtime
ARG UBUNTU_VERSION=22.04
ARG TORCH_VERSION=2.5.1
ARG TORCH_CUDA_TAG=cu124

# ============================================================================
# Builder: python venv with torch, Spark-TTS runtime deps, and the API layer.
# ============================================================================
FROM nvidia/cuda:${CUDA_VERSION}-${CUDA_FLAVOR}-ubuntu${UBUNTU_VERSION} AS builder

ARG TORCH_VERSION
ARG TORCH_CUDA_TAG

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-venv \
        python3.10-dev \
        python3-pip \
        build-essential \
        git \
        ca-certificates \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
RUN pip install --upgrade pip

# PyTorch — single layer, matched to Spark-TTS' torch==2.5.1 constraint.
RUN pip install --index-url https://download.pytorch.org/whl/${TORCH_CUDA_TAG} \
        torch==${TORCH_VERSION} torchaudio==${TORCH_VERSION}

# API layer dependencies.
COPY docker/requirements.api.txt /tmp/requirements.api.txt
RUN pip install -r /tmp/requirements.api.txt

# Spark-TTS engine dependencies (curated subset of upstream requirements.txt).
COPY docker/requirements.engine.txt /tmp/requirements.engine.txt
RUN pip install -r /tmp/requirements.engine.txt

# hf_transfer accelerates model weight downloads via HF Hub.
RUN pip install hf_transfer

# Strip bytecode caches the builds left behind.
RUN find /opt/venv -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true

# ============================================================================
# Runtime: minimal image carrying only the venv + Spark-TTS source + app/.
# ============================================================================
FROM nvidia/cuda:${CUDA_VERSION}-${CUDA_FLAVOR}-ubuntu${UBUNTU_VERSION}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    HF_HOME=/root/.cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/opt/api:/opt/api/engine \
    SPARKTTS_ROOT=/opt/api/engine

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        ffmpeg \
        libsndfile1 \
        ca-certificates \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /opt/api

# The engine submodule (Spark-TTS upstream) is consumed via PYTHONPATH —
# upstream has no pyproject.toml/setup.py, so we do not pip install it.
COPY engine /opt/api/engine
COPY app /opt/api/app
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

VOLUME ["/voices", "/root/.cache"]
EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
