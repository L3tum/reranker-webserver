#!/bin/bash
# Quick test script for the reranker API
# Usage: ./test_rerank.sh [BASE_URL] [API_KEY]

BASE_URL="${1:-http://localhost:8000}"
API_KEY="${2:-}"

echo "=== Health Check ==="
curl -s "${BASE_URL}/health" | python3 -m json.tool
echo

echo "=== List Models ==="
curl -s "${BASE_URL}/v1/models" | python3 -m json.tool
echo

AUTH_HEADER=""
if [ -n "$API_KEY" ]; then
    AUTH_HEADER="-H \"Authorization: Bearer ${API_KEY}\""
fi

echo "=== Rerank Test ==="
curl -s "${BASE_URL}/v1/rerank" \
  -H "Content-Type: application/json" \
  ${AUTH_HEADER} \
  -d '{
    "model": "cross-encoder/ettin-reranker-400m-v1",
    "query": "How do I run an Ettin reranker for Open-WebUI?",
    "documents": [
      "Open-WebUI can use an external reranker endpoint.",
      "Docker image pruning frees unused image layers.",
      "Ettin reranker requires a recent SentenceTransformers version."
    ],
    "top_n": 2
  }' | python3 -m json.tool
echo

echo "Expected: documents 0 and 2 should rank above document 1 (Docker pruning)"
