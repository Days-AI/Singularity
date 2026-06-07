"""LangChain tool layer for evidence collection (Phase 1).

This package wraps every external data source (yfinance, Google Trends, Serper,
DuckDuckGo, Wikipedia, arXiv, Parallel web) behind a uniform async fetcher and,
when LangChain is installed, exposes them as LangChain ``StructuredTool`` objects
for the CrewAI synthesis layer (Phase 2).

Everything is import-guarded: if ``langchain`` is missing the StructuredTool
builders simply return ``None`` and the rest of the system keeps working through
the plain async fetchers and the hand-rolled fallback in ``agents.evidence``.
"""
from __future__ import annotations

from tools.registry import (
    LANGCHAIN_AVAILABLE,
    SOURCE_FETCHERS,
    collect_via_tools,
)

__all__ = ["LANGCHAIN_AVAILABLE", "SOURCE_FETCHERS", "collect_via_tools"]
