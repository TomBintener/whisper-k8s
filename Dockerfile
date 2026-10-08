# A multi-platform Dockerfile that supports both amd64 (CUDA) and arm64 (CPU) using a multi-stage build for clarity and caching.

# ==================================================================
# Base stage with common OS packages that rarely change
# ==================================================================
FROM python:3.11-slim AS base
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg ca-certificates curl tini openssh-client \
 && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --upgrade pip

# ==================================================================
# Final image, combining all dependencies into a single, efficient layer
# ==================================================================
FROM base
ARG TARGETPLATFORM

# Set ENV vars early. Use ${VAR:-} to avoid undefined variable errors.
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PORT=8080 SERVICE=bridge \
    LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}

# Install all Python dependencies in a single RUN command to optimize layer caching and atomicity.
RUN \
    if [ "$TARGETPLATFORM" = "linux/amd64" ]; then \
        echo "--- Installing for linux/amd64 (CUDA) ---"; \
        pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121; \
        pip install "faster-whisper[cuda]==1.2.1"; \
        pip install "nvidia-cudnn-cu12==9.1.0.70"; \
    else \
        echo "--- Installing for $TARGETPLATFORM (CPU) ---"; \
        pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu; \
        pip install "faster-whisper==1.2.1"; \
    fi \
    # Install common dependencies for all platforms
    && pip install \
        openai-whisper==20250625 \
        kubernetes==29.0.0 \
        httpx==0.27.2 \
        fastapi==0.115.0 \
        uvicorn[standard]==0.30.6 \
        pydantic==2.9.2 \
        "redis>=5.0.0"

WORKDIR /app
COPY app/ .
EXPOSE 8080
ENTRYPOINT ["/usr/bin/tini","--"]
CMD ["python","/app/entry.py"]
