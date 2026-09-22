"""Supabase pgvector store for evidence + report chunks.

Uses the Supabase service client to upsert embedded chunks into the
`evidence_chunks` table and to run cosine similarity search via the
`match_documents` RPC (see db/migrations/002_rag.sql).

Import/credential-guarded: `available()` is only True when RAG is enabled,
Supabase is configured, and local embeddings loaded.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from config import get_settings
from rag import embeddings

logger = logging.getLogger("singularity.rag.vectorstore")

_client = None
_client_tried = False
_schema_ok: bool | None = None


def _schema_unavailable(exc: Exception) -> bool:
    """True when Supabase RAG tables/RPCs are missing on the remote project."""
    msg = str(exc).lower()
    return "evidence_chunks" in msg or "match_documents" in msg or "pgrst202" in msg or "pgrst205" in msg


def _mark_schema_unavailable(exc: Exception) -> None:
    global _schema_ok
    if _schema_unavailable(exc):
        _schema_ok = False
        logger.warning(
            "RAG schema unavailable on Supabase (apply db/migrations/002_rag.sql); "
            "disabling vector retrieval for this process."
        )


def _get_client():
    global _client, _client_tried
    if _client_tried:
        return _client
    _client_tried = True
    settings = get_settings()
    if not settings.supabase_enabled:
        return None
    try:
        from supabase import create_client

        _client = create_client(settings.supabase_url, settings.supabase_service_key)
    except Exception as exc:  # noqa: BLE001
        logger.warning("RAG Supabase client init failed: %s", exc)
        _client = None
    return _client


def available() -> bool:
    settings = get_settings()
    if _schema_ok is False:
        return False
    return bool(
        settings.rag_enabled
        and _get_client() is not None
        and embeddings.embeddings_available()
    )


async def upsert_chunks(chunks: list[dict[str, Any]]) -> int:
    """Embed and insert chunks. Each chunk: {content, metadata, session_id}."""
    if not available() or not chunks:
        return 0
    client = _get_client()
    contents = [str(c.get("content", "")) for c in chunks]
    vectors = await embeddings.embed_texts(contents)
    if not vectors:
        return 0
    rows = []
    for chunk, vec in zip(chunks, vectors):
        rows.append(
            {
                "session_id": chunk.get("session_id"),
                "content": chunk.get("content"),
                "metadata": chunk.get("metadata") or {},
                "embedding": vec,
            }
        )

    def _insert() -> int:
        try:
            client.table("evidence_chunks").insert(rows).execute()
            return len(rows)
        except Exception as exc:  # noqa: BLE001
            _mark_schema_unavailable(exc)
            logger.warning("evidence_chunks insert failed: %s", exc)
            return 0

    return await asyncio.to_thread(_insert)


async def similarity_search(query: str, k: int = 5) -> list[dict[str, Any]]:
    """Return up to k matched chunks: [{content, metadata, similarity}]."""
    if not available():
        return []
    client = _get_client()
    vec = await embeddings.embed_query(query)
    if not vec:
        return []

    def _search() -> list[dict[str, Any]]:
        last_exc: Exception | None = None
        # Remote projects may use filter_type (text) instead of filter (jsonb).
        payloads = (
            {"query_embedding": vec, "match_count": k, "filter_type": None},
            {"query_embedding": vec, "match_count": k, "filter_type": ""},
            {"query_embedding": vec, "match_count": k, "filter": {}},
        )
        for params in payloads:
            try:
                res = client.rpc("match_documents", params).execute()
                global _schema_ok
                _schema_ok = True
                return res.data or []
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
        if last_exc:
            _mark_schema_unavailable(last_exc)
            logger.warning("match_documents rpc failed: %s", last_exc)
        return []

    return await asyncio.to_thread(_search)
