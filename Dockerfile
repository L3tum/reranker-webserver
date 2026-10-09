FROM nvidia/cuda:13.3.0-runtime-ubi9@sha256:c4714db36d49c2169a8ce53a423b30a43cbf9f51d59ba15c4203ada1a42029d8

LABEL org.opencontainers.image.source="https://forgejo.mortimer.website/l3tum/reranker-webserver"
LABEL org.opencontainers.image.description="OpenAI-compatible reranker server for Ettin cross-encoder models"

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Install Python 3.11 from AppStream + runtime dependencies
# UBI9 base includes Python 3.9, AppStream provides 3.11
RUN dnf install -y --setopt=install_weak_deps=False \
    python3.11 \
    python3.11-pip \
    ca-certificates \
  && dnf clean all

# Create virtual environment to avoid system package conflicts
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip inside venv (no system conflicts)
RUN pip install --upgrade pip

# PyTorch with CUDA 13.2 support (forward-compatible with CUDA 13.3 runtime)
RUN pip install torch --index-url https://download.pytorch.org/whl/cu132

# Application dependencies - sentence-transformers pulls compatible transformers
RUN pip install \
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
