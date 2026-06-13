FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

LABEL org.opencontainers.image.source="https://forgejo.mortimer.website/l3tum/reranker-webserver"
LABEL org.opencontainers.image.description="OpenAI-compatible reranker server for Ettin cross-encoder models"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --break-system-packages --upgrade pip

# PyTorch with CUDA 12.8 support
RUN python3 -m pip install --break-system-packages \
    torch --index-url https://download.pytorch.org/whl/cu128

# Application dependencies - sentence-transformers pulls compatible transformers
RUN python3 -m pip install --break-system-packages \
    "sentence-transformers>=5.4.1" \
    fastapi \
    "uvicorn[standard]" \
    numpy

WORKDIR /app
COPY .dockerignore .
COPY server.py /app/server.py

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["python3", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
