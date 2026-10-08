# Reranker Webserver

A lightweight, OpenAI-compatible reranking server built on `sentence-transformers` and FastAPI. Designed to work with [Open-WebUI](https://github.com/open-webui/open-webui) via its external reranker integration.

Primary model: **[cross-encoder/ettin-reranker-400m-v1](https://huggingface.co/cross-encoder/ettin-reranker-400m-v1)**

## Why this approach?

- **Infinity** is blocked by older `transformers` dependency support ([issue #657](https://github.com/michaelfeil/infinity/issues/657))
- **ONNX/optimum-onnx** has open `transformers>=5.0.0` compatibility issues ([issue #114](https://github.com/huggingface/optimum-onx/issues/114))
- A custom `sentence-transformers` server is the most direct and stable route today

## Quick Start

```bash
# Build
docker build -t local/ettin-reranker:latest .

# Run
docker run --rm \
  --gpus all \
  -p 8000:8000 \
  -e MODEL_ID=cross-encoder/ettin-reranker-400m-v1 \
  -e DEVICE=cuda \
  local/ettin-reranker:latest
```

Or with docker-compose:

```bash
docker compose up --build
```

## API

### Health Check

```
GET /health
```

Returns: `{"status": "ok", "model": "..."}`

### List Models

```
GET /v1/models
```

Returns OpenAI-compatible model list.

### Rerank

```
POST /v1/rerank
```

**Request body:**

```json
{
  "model": "cross-encoder/ettin-reranker-400m-v1",
  "query": "How do I use a reranker?",
  "documents": [
    "Rerankers score query-document pairs.",
    "Docker builds container images.",
    "Ettin is a cross-encoder model."
  ],
  "top_n": 2,
  "return_documents": true
}
```

**Response:**

```json
{
  "object": "rerank",
  "model": "cross-encoder/ettin-reranker-400m-v1",
  "results": [
    {"index": 0, "relevance_score": 0.95, "document": "..."},
    {"index": 2, "relevance_score": 0.72, "document": "..."}
  ]
}
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_ID` | `cross-encoder/ettin-reranker-400m-v1` | HuggingFace model to load |
| `DEVICE` | `cuda` | Device to run on (`cuda` or `cpu`) |
| `API_KEY` | *(empty)* | Bearer token for auth (leave empty to disable) |
| `BATCH_SIZE` | `16` | Inference batch size |
| `HOST` | `0.0.0.0` | Listen address |
| `PORT` | `8000` | Listen port |
| `HF_HOME` | `~/.cache/huggingface` | HuggingFace cache directory |
| `HF_HUB_OFFLINE` | `0` | Set to `1` to force offline mode |
| `TRANSFORMERS_OFFLINE` | `0` | Set to `1` to force offline mode |

## Integration with Open-WebUI

Set these environment variables in Open-WebUI:

```yaml
environment:
  RAG_RERANKING_ENGINE: external
  RAG_RERANKING_MODEL: cross-encoder/ettin-reranker-400m-v1
  RAG_EXTERNAL_RERANKER_URL: http://reranker:8000/v1/rerank
  RAG_EXTERNAL_RERANKER_API_KEY: your-secret-key
  RAG_EXTERNAL_RERANKER_TIMEOUT: "60"
```

See the [Open-WebUI docs](https://docs.openwebui.com/reference/env-configuration/) for details.

## Pre-downloading the Model (Offline Mode)

To use the server offline, pre-download the model into the HuggingFace cache.
Both steps use `HF_HOME=/app/.cache` so the cache lands in the mounted volume
(the default `HF_HOME` is `~/.cache/huggingface`):

```bash
docker run --rm \
  -v /path/to/cache:/app/.cache \
  -e HF_HOME=/app/.cache \
  python:3.12-slim \
  sh -lc '
    pip install -U huggingface_hub[hf_transfer] &&
    HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download \
      cross-encoder/ettin-reranker-400m-v1
  '
```

Then mount the same cache into the server and force offline mode:

```bash
docker run --rm \
  --gpus all \
  -p 8000:8000 \
  -v /path/to/cache:/app/.cache \
  -e HF_HOME=/app/.cache \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e MODEL_ID=cross-encoder/ettin-reranker-400m-v1 \
  -e DEVICE=cuda \
  local/ettin-reranker:latest
```

## Testing

```bash
curl -s http://localhost:8000/v1/rerank \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-key" \
  -d '{
    "model": "cross-encoder/ettin-reranker-400m-v1",
    "query": "How do I run an Ettin reranker for Open-WebUI?",
    "documents": [
      "Open-WebUI can use an external reranker endpoint.",
      "Docker image pruning frees unused image layers.",
      "Ettin reranker requires a recent SentenceTransformers version."
    ],
    "top_n": 2
  }' | jq
```

## Model Variants

| Model | Size | ENV |
|---|---|---|
| Ettin 150M | ~150M params | `MODEL_ID=cross-encoder/ettin-reranker-150m-v1` |
| **Ettin 400M** | **~400M params** | **`MODEL_ID=cross-encoder/ettin-reranker-400m-v1`** |

## License

MIT
