"""Shared sentence-transformer access (delegates to rag.embeddings)."""
from __future__ import annotations

from rag import embeddings as _rag_embeddings


def embeddings_available() -> bool:
    return _rag_embeddings.embeddings_available()


async def embed_texts(texts: list[str]) -> list[list[float]]:
    return await _rag_embeddings.embed_texts(texts)


async def embed_query(text: str) -> list[float]:
    return await _rag_embeddings.embed_query(text)
