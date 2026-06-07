"""Tool registry: route DAG nodes to source fetchers and aggregate results.

`collect_via_tools` is the LangChain-flavored replacement for the hand-rolled
routing in `agents.evidence.collect`. It selects a set of sources based on the
node's `agent_type` and task semantics, runs them concurrently, and merges the
items into a single `EvidenceResult` (deduplicated, capped, confidence-weighted).

When LangChain is installed, `build_langchain_tools()` exposes the same sources
as `StructuredTool` objects so the CrewAI agents in Phase 2 can call them.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

from config import get_settings
from state import DagNode, EvidenceItem

from tools import sources

logger = logging.getLogger("singularity.tools.registry")

try:  # LangChain is optional; only needed to expose tools to CrewAI.
    from langchain_core.tools import StructuredTool  # type: ignore

    LANGCHAIN_AVAILABLE = True
except Exception:  # noqa: BLE001
    StructuredTool = None  # type: ignore
    LANGCHAIN_AVAILABLE = False


FetchFn = Callable[..., Awaitable[sources.SourceResult]]

# Canonical fetcher registry (name -> async fetcher).
SOURCE_FETCHERS: dict[str, FetchFn] = {
    "financial": sources.fetch_financial,
    "serper": sources.fetch_serper,
    "parallel": sources.fetch_parallel,
    "arxiv": sources.fetch_arxiv,
    "wikipedia": sources.fetch_wikipedia,
    "duckduckgo": sources.fetch_ddg,
    "gdelt": sources.fetch_gdelt,
}

_RESEARCH_KEYWORDS = ("research", "study", "paper", "academic", "scientific", "evidence", "literature")

_MAX_ITEMS = 8


def _select_sources(node: DagNode) -> list[str]:
    """Choose which named sources to run for a node."""
    task = (node.task or "").lower()
    settings = get_settings()
    if node.agent_type == "financial":
        return ["financial"]

    selected: list[str] = []
    # Premium web search first (keyed), then keyless fallbacks for breadth.
    if settings.parallel_enabled:
        selected.append("parallel")
    if settings.serper_api_key:
        selected.append("serper")
    selected.append("duckduckgo")
    selected.append("wikipedia")
    # GDELT global news/event search joins routing only once explicitly enabled.
    if settings.gdelt_enabled:
        selected.append("gdelt")
    # Augment with academic literature for research-flavored tasks.
    if any(k in task for k in _RESEARCH_KEYWORDS):
        selected.append("arxiv")
    return selected


async def collect_via_tools(node: DagNode, query: str, web_sources_enabled: bool = True):
    """Run the selected sources for a node and merge into an EvidenceResult."""
    # Imported lazily to avoid an import cycle (evidence imports this module).
    from agents.evidence import EvidenceResult

    if not web_sources_enabled and node.agent_type != "financial":
        return EvidenceResult(
            items=[
                EvidenceItem(
                    source="Offline",
                    title=node.task,
                    detail="Web sources disabled; proceeding on priors.",
                )
            ],
            confidence=0.3,
        )

    names = _select_sources(node)
    search_q = query if node.agent_type == "financial" else f"{query} {node.task}".strip()

    fetchers = [SOURCE_FETCHERS[name](search_q, web_sources_enabled=web_sources_enabled) for name in names]
    results = await asyncio.gather(*fetchers, return_exceptions=True)

    merged_items: list[EvidenceItem] = []
    merged_series = []
    confidences: list[float] = []
    seen_titles: set[str] = set()
    for name, res in zip(names, results):
        if isinstance(res, Exception):
            logger.warning("source %s raised: %s", name, res)
            continue
        if res.confidence > 0:
            confidences.append(res.confidence)
        for item in res.items:
            key = (item.title or "").strip().lower()
            if key and key in seen_titles:
                continue
            seen_titles.add(key)
            merged_items.append(item)
        merged_series.extend(res.series)

    if not merged_items:
        # All live sources empty/failed - degrade to the hand-rolled web fallback.
        from agents import evidence as evidence_agent

        return await evidence_agent._web(node, query, web_sources_enabled)  # noqa: SLF001

    merged_items = merged_items[:_MAX_ITEMS]
    confidence = max(confidences) if confidences else 0.5
    return EvidenceResult(items=merged_items, series=merged_series, confidence=confidence)


# --- LangChain StructuredTool exposure (for CrewAI agents, Phase 2) ----------
def _render_items(items: list[EvidenceItem]) -> str:
    if not items:
        return "No results."
    lines = []
    for it in items[:6]:
        url = f" ({it.url})" if it.url else ""
        lines.append(f"- [{it.source}] {it.title}: {it.detail}{url}")
    return "\n".join(lines)


def build_langchain_tools(web_sources_enabled: bool = True) -> list:
    """Return LangChain StructuredTools wrapping the web/research sources.

    Returns an empty list when LangChain is not installed.
    """
    if not LANGCHAIN_AVAILABLE:
        return []

    def _make(name: str, description: str):
        fetcher = SOURCE_FETCHERS[name]

        async def _run(query: str) -> str:
            res = await fetcher(query, web_sources_enabled=web_sources_enabled)
            return _render_items(res.items)

        return StructuredTool.from_function(
            coroutine=_run,
            name=f"{name}_search",
            description=description,
        )

    specs = {
        "parallel": "Search the live web via the Parallel API for current facts and articles.",
        "serper": "Search Google (via Serper) for recent web results.",
        "duckduckgo": "Search the web via DuckDuckGo (keyless fallback).",
        "wikipedia": "Look up encyclopedic background on a topic.",
        "arxiv": "Find recent academic papers and research abstracts.",
        "gdelt": "Search GDELT for recent global news articles on a topic.",
    }
    tools = []
    settings = get_settings()
    for name, desc in specs.items():
        if name == "parallel" and not settings.parallel_enabled:
            continue
        if name == "serper" and not settings.serper_api_key:
            continue
        if name == "gdelt" and not settings.gdelt_enabled:
            continue
        tools.append(_make(name, desc))
    return tools
