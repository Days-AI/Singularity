"""Local HuggingFace sentence-transformers embeddings (CPU, lazy singleton).

Import-guarded: if `sentence-transformers` is not installed, `embeddings_available()`
returns False and callers skip embedding rather than crashing.
"""
from __future__ import annotations

import asyncio
import logging
import threading

from config import get_settings

logger = logging.getLogger("singularity.rag.embeddings")

_model = None
_load_lock = threading.Lock()
_load_failed = False


def _load_model():
    global _model, _load_failed
    if _model is not None or _load_failed:
        return _model
    with _load_lock:
        if _model is not None or _load_failed:
            return _model
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            settings = get_settings()
            logger.info("Loading embedding model %s", settings.embedding_model_id)
            _model = SentenceTransformer(settings.embedding_model_id, device="cpu")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Embedding model load failed (%s); RAG disabled", exc)
            _load_failed = True
            _model = None
    return _model


def embeddings_available() -> bool:
    return _load_model() is not None


def _embed_blocking(texts: list[str]) -> list[list[float]]:
    model = _load_model()
    if model is None:
        return []
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [[float(x) for x in v] for v in vecs]


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts; returns [] if embeddings are unavailable."""
    if not texts or not embeddings_available():
        return []
    return await asyncio.to_thread(_embed_blocking, texts)


async def embed_query(text: str) -> list[float]:
    vecs = await embed_texts([text])
    return vecs[0] if vecs else []
