"""GraphRAG retrieval: vector hits augmented with causal-graph neighbors.

Plain vector search retrieves semantically similar prior evidence/report chunks.
The "graph" step expands context by finding causal nodes mentioned in the hits
and pulling their 1-hop neighbors from the run's causal graph, then optionally
seeding a second vector search with the strongest neighbor driver. The result is
a compact context string fed into report/crew synthesis via
`state.metrics["rag_context"]`.
"""
from __future__ import annotations

import logging

from config import get_settings
from rag import vectorstore
from state import SingularityState

logger = logging.getLogger("singularity.rag.retriever")

_MAX_CONTEXT_CHARS = 1800


def _causal_adjacency(state: SingularityState) -> tuple[dict[str, str], dict[str, set[str]]]:
    """Return (node_id -> label, label -> set(neighbor labels))."""
    labels: dict[str, str] = {}
    adj: dict[str, set[str]] = {}
    if not state.causal:
        return labels, adj
    labels = {n.id: n.label for n in state.causal.nodes}
    for e in state.causal.edges:
        a = labels.get(e.source, e.source)
        b = labels.get(e.target, e.target)
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    return labels, adj


async def graph_rag_context(state: SingularityState, query: str | None = None) -> str:
    """Build a RAG context block; returns "" when RAG is unavailable."""
    if not vectorstore.available():
        return ""
    settings = get_settings()
    q = query or state.query
    try:
        hits = await vectorstore.similarity_search(q, k=settings.rag_top_k)
    except Exception as exc:  # noqa: BLE001
        logger.warning("similarity_search failed: %s", exc)
        return ""
    if not hits:
        return ""

    labels, adj = _causal_adjacency(state)
    # Graph expansion: which causal drivers are mentioned in the hits?
    mentioned: set[str] = set()
    for hit in hits:
        content = str(hit.get("content", "")).lower()
        for label in adj:
            if label and label.lower() in content:
                mentioned.add(label)
    neighbors: set[str] = set()
    for label in mentioned:
        neighbors |= adj.get(label, set())
    neighbors -= mentioned

    # Seed a second retrieval from the strongest related driver for depth.
    if neighbors:
        seed = sorted(neighbors)[0]
        try:
            extra = await vectorstore.similarity_search(seed, k=2)
            seen = {str(h.get("content")) for h in hits}
            for h in extra:
                if str(h.get("content")) not in seen:
                    hits.append(h)
        except Exception:  # noqa: BLE001
            pass

    lines: list[str] = ["Relevant prior context (retrieved):"]
    for hit in hits[: settings.rag_top_k + 2]:
        meta = hit.get("metadata") or {}
        tag = meta.get("kind", "doc")
        snippet = str(hit.get("content", "")).strip().replace("\n", " ")
        if snippet:
            lines.append(f"- [{tag}] {snippet[:240]}")
    if neighbors:
        lines.append("Related causal drivers: " + ", ".join(sorted(neighbors)[:6]))

    context = "\n".join(lines)
    return context[:_MAX_CONTEXT_CHARS]
