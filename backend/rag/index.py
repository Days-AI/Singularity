"""Ingest a completed run's evidence + report sections into the vector store."""
from __future__ import annotations

import logging

from rag import vectorstore
from state import SingularityState

logger = logging.getLogger("singularity.rag.index")


async def index_run(state: SingularityState) -> int:
    """Embed and persist evidence items and report sections for future RAG.

    No-op (returns 0) when RAG is disabled/unavailable. Never raises.
    """
    if not vectorstore.available():
        return 0
    session_id = state.session_id or state.flow_uuid
    chunks: list[dict] = []
    for ev in state.evidence:
        text = f"{ev.title}. {ev.detail}".strip()
        if not text:
            continue
        chunks.append(
            {
                "session_id": session_id,
                "content": text,
                "metadata": {
                    "kind": "evidence",
                    "source": ev.source,
                    "query": state.query,
                    "url": ev.url,
                },
            }
        )
    for sec in state.report_sections:
        if not (sec.content or "").strip():
            continue
        chunks.append(
            {
                "session_id": session_id,
                "content": f"{sec.section}: {sec.content}",
                "metadata": {
                    "kind": "report",
                    "section": sec.section,
                    "query": state.query,
                },
            }
        )
    try:
        n = await vectorstore.upsert_chunks(chunks)
        if n:
            logger.info("Indexed %d RAG chunks for session %s", n, session_id)
        return n
    except Exception as exc:  # noqa: BLE001
        logger.warning("index_run failed: %s", exc)
        return 0
