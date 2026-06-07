"""Graph-RAG layer (Phase 1).

Embeds evidence items and report sections into Supabase pgvector and retrieves
relevant prior context for synthesis. "Graph" RAG augments plain vector hits with
neighbors of matched nodes in the run's causal graph.

All components are import-guarded and flag-gated (`RAG_ENABLED`): if
sentence-transformers or Supabase is unavailable, retrieval returns empty context
and the pipeline is unaffected.
"""
from __future__ import annotations

from rag.retriever import graph_rag_context
from rag.index import index_run

__all__ = ["graph_rag_context", "index_run"]
