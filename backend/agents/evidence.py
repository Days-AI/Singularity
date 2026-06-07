"""Evidence-grounding agents (spec section 5 / Layer 2).

Each evidence base_root is routed to a real data source based on its agent_type
and task text:
  - financial  -> yfinance (price/volume series)
  - web_search -> Serper (if key) or DuckDuckGo; pytrends for search velocity;
                  Wikipedia for structured background.

Every source is wrapped so a network/library failure degrades to a deterministic
synthetic series rather than breaking the flow. All blocking library calls run
in a worker thread to keep the event loop responsive. The agents also assemble
numeric time-series the causal and forecast engines consume.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import logging
import math
from dataclasses import dataclass, field

from config import get_settings
from state import DagNode, EvidenceItem, TimeSeries

logger = logging.getLogger("singularity.evidence")

_HORIZON_HISTORY = 180  # days of history we assemble for downstream engines


@dataclass
class EvidenceResult:
    items: list[EvidenceItem] = field(default_factory=list)
    series: list[TimeSeries] = field(default_factory=list)
    confidence: float = 0.6


async def collect(node: DagNode, query: str) -> EvidenceResult:
    task = node.task.lower()
    if node.agent_type == "financial":
        return await _financial(node, query)
    # web_search family - choose by task semantics.
    if any(k in task for k in ("trend", "search velocity", "social", "sentiment", "interest")):
        return await _trends(node, query)
    return await _web(node, query)


# --- financial ---------------------------------------------------------------
async def _financial(node: DagNode, query: str) -> EvidenceResult:
    ticker = _ticker_for(query)
    try:
        items, series = await asyncio.to_thread(_yfinance_blocking, ticker)
        if series.values:
            return EvidenceResult(items=items, series=[series], confidence=0.82)
    except Exception as exc:  # noqa: BLE001
        logger.warning("yfinance failed (%s): %s", ticker, exc)
    series = _synthetic_series(f"{ticker} Close", base=120.0, drift=0.15, vol=2.5, seed=ticker)
    items = [
        EvidenceItem(source="Synthetic-Macro", title=f"{ticker} proxy series",
                     detail="Live market data unavailable; using deterministic macro proxy.",
                     value=series.values[-1], unit="INR"),
    ]
    return EvidenceResult(items=items, series=[series], confidence=0.4)


def _yfinance_blocking(ticker: str) -> tuple[list[EvidenceItem], TimeSeries]:
    import yfinance as yf

    hist = yf.Ticker(ticker).history(period="6mo", interval="1d")
    if hist is None or hist.empty:
        raise ValueError("no history")
    closes = hist["Close"].dropna()
    dates = [d.strftime("%Y-%m-%d") for d in closes.index.to_pydatetime()]
    values = [float(v) for v in closes.values]
    last = values[-1]
    first = values[0]
    pct = (last - first) / first * 100 if first else 0.0
    items = [
        EvidenceItem(source="yFinance", title=f"{ticker} 6-month performance",
                     detail=f"Close moved {pct:+.1f}% over 6 months.", value=round(last, 2),
                     unit="INR"),
        EvidenceItem(source="yFinance", title=f"{ticker} volatility",
                     detail=f"Latest close {last:.2f}, range {min(values):.1f}-{max(values):.1f}.",
                     value=round(_stdev(values), 2)),
    ]
    return items, TimeSeries(name=f"{ticker} Close", dates=dates, values=values)


def _ticker_for(query: str) -> str:
    q = query.lower()
    if "india" in q and ("ev" in q or "electric" in q or "auto" in q or "vehicle" in q):
        return "TATAMOTORS.NS"
    if "india" in q:
        return "^NSEI"
    if "ev" in q or "electric vehicle" in q or "tesla" in q:
        return "TSLA"
    return "^GSPC"


# --- web search --------------------------------------------------------------
async def _web(node: DagNode, query: str) -> EvidenceResult:
    settings = get_settings()
    search_q = f"{query} {node.task}"
    if settings.serper_api_key:
        try:
            items = await _serper(search_q, settings.serper_api_key)
            if items:
                return EvidenceResult(items=items, series=[], confidence=0.78)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Serper failed: %s", exc)
    try:
        items = await asyncio.to_thread(_ddg_blocking, search_q)
        if items:
            return EvidenceResult(items=items, series=[], confidence=0.7)
    except Exception as exc:  # noqa: BLE001
        logger.warning("DuckDuckGo failed: %s", exc)
    try:
        items = await asyncio.to_thread(_wiki_blocking, query)
        if items:
            return EvidenceResult(items=items, series=[], confidence=0.55)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Wikipedia failed: %s", exc)
    return EvidenceResult(
        items=[EvidenceItem(source="Heuristic", title=node.task,
                            detail="No live web source reachable; proceeding on priors.")],
        confidence=0.3,
    )


async def _serper(query: str, api_key: str) -> list[EvidenceItem]:
    import httpx

    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json={"q": query, "num": 6},
        )
        resp.raise_for_status()
        data = resp.json()
    items: list[EvidenceItem] = []
    for r in (data.get("organic") or [])[:6]:
        items.append(
            EvidenceItem(source="Serper", title=str(r.get("title", ""))[:140],
                         detail=str(r.get("snippet", ""))[:240], url=r.get("link"))
        )
    return items


def _ddg_blocking(query: str) -> list[EvidenceItem]:
    from ddgs import DDGS

    items: list[EvidenceItem] = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=6):
            items.append(
                EvidenceItem(source="DuckDuckGo", title=str(r.get("title", ""))[:140],
                             detail=str(r.get("body", ""))[:240], url=r.get("href"))
            )
    return items


def _wiki_blocking(query: str) -> list[EvidenceItem]:
    import wikipedia

    terms = wikipedia.search(query, results=2)
    items: list[EvidenceItem] = []
    for term in terms:
        try:
            summary = wikipedia.summary(term, sentences=2, auto_suggest=False)
        except Exception:  # noqa: BLE001
            continue
        items.append(EvidenceItem(source="Wikipedia", title=term, detail=summary[:260]))
    return items


# --- trends / search velocity ------------------------------------------------
async def _trends(node: DagNode, query: str) -> EvidenceResult:
    kw = _keyword_for(query)
    try:
        items, series = await asyncio.to_thread(_pytrends_blocking, kw)
        if series.values:
            return EvidenceResult(items=items, series=[series], confidence=0.72)
    except Exception as exc:  # noqa: BLE001
        logger.warning("pytrends failed: %s", exc)
    series = _synthetic_series(f"{kw} search interest", base=55.0, drift=0.25, vol=6.0, seed=kw)
    items = [EvidenceItem(source="Synthetic-Trends", title=f"{kw} interest proxy",
                          detail="Google Trends unavailable; using deterministic interest proxy.",
                          value=series.values[-1], unit="index")]
    return EvidenceResult(items=items, series=[series], confidence=0.35)


def _pytrends_blocking(keyword: str) -> tuple[list[EvidenceItem], TimeSeries]:
    from pytrends.request import TrendReq

    py = TrendReq(hl="en-US", tz=330)
    py.build_payload([keyword], timeframe="today 6-m")
    df = py.interest_over_time()
    if df is None or df.empty:
        raise ValueError("no trends data")
    col = df[keyword].dropna()
    dates = [d.strftime("%Y-%m-%d") for d in col.index.to_pydatetime()]
    values = [float(v) for v in col.values]
    items = [EvidenceItem(source="GoogleTrends", title=f"{keyword} search velocity",
                          detail=f"Mean interest {sum(values)/len(values):.0f}, latest {values[-1]:.0f}.",
                          value=round(values[-1], 1), unit="index")]
    return items, TimeSeries(name=f"{keyword} search interest", dates=dates, values=values)


def _keyword_for(query: str) -> str:
    q = query.lower()
    if "ev" in q or "electric" in q:
        return "electric vehicle"
    words = [w for w in query.split() if len(w) > 3]
    return words[0] if words else "market"


# --- deterministic fallback --------------------------------------------------
def _synthetic_series(name: str, base: float, drift: float, vol: float, seed: str) -> TimeSeries:
    """Reproducible series so causal/forecast always have aligned data."""
    n = _HORIZON_HISTORY
    h = int(hashlib.sha256(seed.encode()).hexdigest(), 16)
    today = dt.date.today()
    dates: list[str] = []
    values: list[float] = []
    val = base
    for i in range(n):
        h = (1103515245 * h + 12345) & 0x7FFFFFFF
        noise = ((h % 1000) / 1000.0 - 0.5) * vol
        seasonal = math.sin(2 * math.pi * i / 30.0) * vol * 0.4
        val = max(1.0, val + drift + noise + seasonal * 0.1)
        dates.append((today - dt.timedelta(days=(n - i))).strftime("%Y-%m-%d"))
        values.append(round(val, 3))
    return TimeSeries(name=name, dates=dates, values=values)


def _stdev(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
