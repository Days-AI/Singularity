"""Async source fetchers behind the LangChain tool layer.

Each fetcher returns a ``SourceResult`` (items + optional time series +
confidence) and degrades gracefully on any failure. The financial / serper /
ddg / wikipedia fetchers reuse the battle-tested blocking helpers in
``agents.evidence`` (imported lazily to avoid an import cycle); ``arxiv`` and
``parallel`` are new here.

All blocking library calls run in a worker thread to keep the event loop free.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from config import get_settings
from state import EvidenceItem, TimeSeries

logger = logging.getLogger("singularity.tools.sources")


@dataclass
class SourceResult:
    items: list[EvidenceItem] = field(default_factory=list)
    series: list[TimeSeries] = field(default_factory=list)
    confidence: float = 0.6


# --- company intelligence (reuse evidence helpers) ---------------------------
async def fetch_financial(query: str, *, web_sources_enabled: bool = True) -> SourceResult:
    """Gemma-orchestrated company intelligence via yfinance (no price series).

    Delegates to ``agents.evidence._company_intel`` so the LangChain tool path and
    the hand-rolled routing path share a single implementation.
    """
    from agents.evidence import _company_intel

    res = await _company_intel(query, web_sources_enabled=web_sources_enabled)
    return SourceResult(items=res.items, series=res.series, confidence=res.confidence)


# --- web sources -------------------------------------------------------------
def _native_tools_enabled() -> bool:
    """True when the official LangChain search wrappers should be used.

    Gated on the LANGCHAIN_NATIVE_TOOLS flag *and* the wrapper packages actually
    being importable, so this degrades transparently to the legacy fetchers.
    """
    if not get_settings().langchain_native_tools:
        return False
    try:
        from tools import search_providers

        return search_providers.LC_COMMUNITY_AVAILABLE or search_providers.LC_PARALLEL_AVAILABLE
    except Exception:  # noqa: BLE001
        return False


async def fetch_serper(query: str, *, web_sources_enabled: bool = True) -> SourceResult:
    settings = get_settings()
    if not web_sources_enabled or not settings.serper_api_key:
        return SourceResult(confidence=0.0)

    if _native_tools_enabled():
        try:
            from tools import search_providers

            raw = await search_providers.aserper_search(query)
            items = search_providers.to_evidence_items(raw, source="Serper")
            if items:
                return SourceResult(items=items, confidence=0.78)
        except Exception as exc:  # noqa: BLE001 - fall back to the legacy path
            logger.warning("Native Serper wrapper failed, using legacy path: %s", exc)

    from agents.evidence import _serper

    try:
        items = await _serper(query, settings.serper_api_key)
        return SourceResult(items=items, confidence=0.78 if items else 0.0)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Serper tool failed: %s", exc)
        return SourceResult(confidence=0.0)


async def fetch_ddg(query: str, *, web_sources_enabled: bool = True) -> SourceResult:
    if not web_sources_enabled:
        return SourceResult(confidence=0.0)

    if _native_tools_enabled():
        try:
            from tools import search_providers

            raw = await search_providers.aduckduckgo_results(query)
            items = search_providers.to_evidence_items(raw, source="DuckDuckGo")
            if items:
                return SourceResult(items=items, confidence=0.7)
        except Exception as exc:  # noqa: BLE001 - fall back to the legacy path
            logger.warning("Native DuckDuckGo wrapper failed, using legacy path: %s", exc)

    from agents.evidence import _ddg_blocking

    try:
        items = await asyncio.to_thread(_ddg_blocking, query)
        return SourceResult(items=items, confidence=0.7 if items else 0.0)
    except Exception as exc:  # noqa: BLE001
        logger.warning("DuckDuckGo tool failed: %s", exc)
        return SourceResult(confidence=0.0)


async def fetch_wikipedia(query: str, *, web_sources_enabled: bool = True) -> SourceResult:
    if not web_sources_enabled:
        return SourceResult(confidence=0.0)
    from agents.evidence import _wiki_blocking

    try:
        items = await asyncio.to_thread(_wiki_blocking, query)
        return SourceResult(items=items, confidence=0.55 if items else 0.0)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Wikipedia tool failed: %s", exc)
        return SourceResult(confidence=0.0)


async def fetch_arxiv(query: str, *, web_sources_enabled: bool = True) -> SourceResult:
    if not web_sources_enabled:
        return SourceResult(confidence=0.0)
    try:
        items = await asyncio.to_thread(_arxiv_blocking, query)
        return SourceResult(items=items, confidence=0.6 if items else 0.0)
    except Exception as exc:  # noqa: BLE001
        logger.warning("arXiv tool failed: %s", exc)
        return SourceResult(confidence=0.0)


def _arxiv_blocking(query: str, max_results: int = 3) -> list[EvidenceItem]:
    import arxiv  # type: ignore

    client = arxiv.Client(page_size=max_results, num_retries=2)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    items: list[EvidenceItem] = []
    for r in client.results(search):
        summary = (r.summary or "").replace("\n", " ").strip()
        items.append(
            EvidenceItem(
                source="arXiv",
                title=str(r.title)[:140],
                detail=summary[:260],
                url=r.entry_id,
            )
        )
    return items


# --- GDELT global news/event search -----------------------------------------
async def fetch_gdelt(query: str, *, web_sources_enabled: bool = True) -> SourceResult:
    """Fetch recent news articles + an article-volume series from GDELT.

    Flag-gated on ``GDELT_ENABLED`` and import-guarded on the optional ``gdeltdoc``
    package, so this returns an empty (zero-confidence) result without side effects
    when GDELT is disabled, unavailable, or fails. The blocking GDELT calls run in a
    worker thread to keep the event loop free, mirroring the other fetchers.
    """
    settings = get_settings()
    if not web_sources_enabled or not settings.gdelt_enabled:
        return SourceResult(confidence=0.0)
    try:
        items, series = await asyncio.to_thread(
            _gdelt_blocking,
            query,
            settings.gdelt_lookback_days,
            settings.gdelt_max_articles,
        )
        return SourceResult(items=items, series=series, confidence=0.7 if items else 0.0)
    except Exception as exc:  # noqa: BLE001
        logger.warning("GDELT tool failed: %s", exc)
        return SourceResult(confidence=0.0)


def _gdelt_query_with_retry(fn, *args, retries: int = 5, wait: float = 10.0):
    """Call ``fn(*args)`` with exponential backoff on GDELT rate-limit errors.

    GDELT's free endpoint is aggressively rate limited. Only ``RateLimitError``
    (matched by class name to avoid importing a private type) is retried; any other
    exception propagates immediately to the caller's degrade-on-failure handler.
    """
    for attempt in range(retries):
        try:
            return fn(*args)
        except Exception as exc:  # noqa: BLE001
            if "RateLimitError" in type(exc).__name__:
                logger.warning(
                    "GDELT rate limited; waiting %.0fs before retry %d/%d",
                    wait,
                    attempt + 1,
                    retries,
                )
                time.sleep(wait)
                wait *= 2  # exponential backoff
            else:
                raise
    raise RuntimeError("GDELT max retries exceeded")


def _gdelt_blocking(
    query: str, lookback_days: int, max_articles: int
) -> tuple[list[EvidenceItem], list[TimeSeries]]:
    """Blocking GDELT fetch: articles -> EvidenceItems, timeline -> TimeSeries.

    Imported lazily so the backend boots without ``gdeltdoc`` installed. Dates use a
    rolling lookback window ending today (UTC) instead of a fixed range.
    """
    from gdeltdoc import Filters, GdeltDoc  # type: ignore

    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=max(1, lookback_days))
    keyword = (query or "").strip()[:200] or "news"
    filters = Filters(
        keyword=keyword,
        start_date=start.isoformat(),
        end_date=end.isoformat(),
    )

    gd = GdeltDoc()
    items: list[EvidenceItem] = []
    series: list[TimeSeries] = []

    articles = _gdelt_query_with_retry(gd.article_search, filters)
    if articles is not None and not articles.empty:
        for _, row in articles.head(max_articles).iterrows():
            title = str(row.get("title") or "").strip()
            if not title:
                continue
            domain = str(row.get("domain") or "").strip()
            seendate = str(row.get("seendate") or "").strip()
            detail = " | ".join(part for part in (domain, seendate) if part) or "GDELT article"
            url = row.get("url")
            items.append(
                EvidenceItem(
                    source="GDELT",
                    title=title[:140],
                    detail=detail[:260],
                    url=str(url) if url else None,
                )
            )

    timeline = _gdelt_query_with_retry(gd.timeline_search, "timelinevol", filters)
    ts = _gdelt_timeline_to_series(timeline)
    if ts is not None:
        series.append(ts)

    return items, series


def _gdelt_timeline_to_series(timeline) -> TimeSeries | None:
    """Convert a GDELT ``timelinevol`` DataFrame into a :class:`TimeSeries`.

    Defensive about column naming across gdeltdoc versions: picks the first
    datetime-like column for dates and the first numeric column for values.
    """
    if timeline is None or getattr(timeline, "empty", True):
        return None

    date_col = next((c for c in timeline.columns if "date" in str(c).lower()), None)
    if date_col is None:
        date_col = timeline.columns[0]

    value_col = None
    for c in timeline.columns:
        if c == date_col:
            continue
        if timeline[c].dtype.kind in "if":  # int / float
            value_col = c
            break
    if value_col is None:
        return None

    dates = [str(d) for d in timeline[date_col].tolist()]
    values = [float(v) for v in timeline[value_col].tolist()]
    if not values:
        return None
    return TimeSeries(name="GDELT article volume", dates=dates, values=values)

async def fetch_parallel(query: str, *, web_sources_enabled: bool = True) -> SourceResult:
    settings = get_settings()
    if not web_sources_enabled or not settings.parallel_api_key:
        return SourceResult(confidence=0.0)

    if _native_tools_enabled():
        try:
            from tools import search_providers

            raw = await search_providers.aparallel_search(query)
            items = search_providers.to_evidence_items(raw, source="Parallel")
            if items:
                return SourceResult(items=items, confidence=0.8)
        except Exception as exc:  # noqa: BLE001 - fall back to the legacy path
            logger.warning("Native Parallel wrapper failed, using legacy path: %s", exc)

    try:
        items = await _parallel_search(query, settings.parallel_api_key, settings.parallel_base_url)
        return SourceResult(items=items, confidence=0.8 if items else 0.0)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Parallel tool failed: %s", exc)
        return SourceResult(confidence=0.0)


async def _parallel_search(query: str, api_key: str, base_url: str) -> list[EvidenceItem]:
    import httpx

    headers = {"x-api-key": api_key, "Content-Type": "application/json"}
    body = {"objective": query, "search_queries": [query], "max_results": 6}
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(f"{base_url.rstrip('/')}/v1beta/search", headers=headers, json=body)
        resp.raise_for_status()
        data = resp.json()
    items: list[EvidenceItem] = []
    # Parallel returns a list of results under "results"; each has url/title and
    # one or more text excerpts. Be defensive about the exact shape.
    for r in (data.get("results") or [])[:6]:
        excerpts = r.get("excerpts") or r.get("content") or []
        if isinstance(excerpts, list):
            detail = " ".join(str(e) for e in excerpts)[:240]
        else:
            detail = str(excerpts)[:240]
        items.append(
            EvidenceItem(
                source="Parallel",
                title=str(r.get("title") or r.get("url") or "")[:140],
                detail=detail,
                url=r.get("url"),
            )
        )
    return items
