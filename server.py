"""
OpenAI-compatible reranking server using sentence-transformers CrossEncoder.

Primary model: cross-encoder/ettin-reranker-400m-v1
Compatible with Open-WebUI's external reranker integration.
"""

import logging
import os
import time
from typing import Any

import numpy as np
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID: str = os.getenv("MODEL_ID", "cross-encoder/ettin-reranker-400m-v1")
DEVICE: str = os.getenv("DEVICE", "cuda")
API_KEY: str = os.getenv("API_KEY", "")
BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "16"))
HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = int(os.getenv("PORT", "8000"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("reranker")

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

logger.info("Loading reranker: %s on %s", MODEL_ID, DEVICE)
model_start = time.time()

model = CrossEncoder(
    MODEL_ID,
    device=DEVICE,
    trust_remote_code=True,
)

model_elapsed = time.time() - model_start
logger.info("Reranker loaded in %.1fs", model_elapsed)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Ettin Reranker",
    description="OpenAI-compatible reranking server for cross-encoder models",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class RerankRequest(BaseModel):
    """Request body for /v1/rerank."""

    model: str | None = None
    query: str = Field(..., min_length=1, description="The query to match against documents")
    documents: list[Any] = Field(..., min_length=1, description="Documents to rerank")
    top_n: int | None = Field(default=None, ge=1, description="Return only top N results")
    return_documents: bool = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check_auth(authorization: str | None) -> None:
    """Validate Bearer token if API_KEY is configured."""
    if not API_KEY:
        return

    expected = f"Bearer {API_KEY}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


def doc_to_text(doc: Any) -> str:
    """Extract text from a document (string, dict, or other)."""
    if isinstance(doc, str):
        return doc

    if isinstance(doc, dict):
        for key in ("text", "content", "document"):
            value = doc.get(key)
            if isinstance(value, str):
                return value

    return str(doc)


def normalize_scores(raw_scores: Any) -> list[float]:
    """
    Normalize CrossEncoder output to a flat list of floats.

    Handles:
    - (N,)          -> regression / single-logit
    - (N, 1)        -> single-logit wrapped
    - (N, 2+)       -> classification logits; use last class as relevance
    """
    scores = np.asarray(raw_scores)

    if scores.ndim == 2:
        if scores.shape[1] == 1:
            scores = scores[:, 0]
        else:
            # Classification: use the positive/relevant class (last column)
            scores = scores[:, -1]

    return [float(x) for x in scores.tolist()]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "model": MODEL_ID}


@app.get("/v1/models")
def models() -> dict[str, Any]:
    """List available models (OpenAI-compatible)."""
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/rerank")
def rerank(
    req: RerankRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    """
    Rerank documents by relevance to a query.

    Returns results sorted by descending relevance score.
    """
    check_auth(authorization)

    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty")

    if len(req.documents) == 0:
        raise HTTPException(status_code=400, detail="Documents list must not be empty")

    # Extract text from documents
    docs_text = [doc_to_text(doc) for doc in req.documents]

    # Build query-document pairs
    pairs = [[req.query, doc] for doc in docs_text]

    # Score
    raw_scores = model.predict(
        pairs,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
    )
    scores = normalize_scores(raw_scores)

    # Rank by score descending
    ranked = sorted(
        enumerate(scores),
        key=lambda item: item[1],
        reverse=True,
    )

    # Apply top_n limit
    if req.top_n is not None:
        ranked = ranked[: req.top_n]

    # Build response
    results = []
    for index, score in ranked:
        item: dict[str, Any] = {
            "index": index,
            "relevance_score": score,
        }
        if req.return_documents:
            item["document"] = req.documents[index]
        results.append(item)

    return {
        "object": "rerank",
        "model": req.model or MODEL_ID,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def internal_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled error: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"},
    )
