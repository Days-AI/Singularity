"""Evidence-grounding agents (spec section 5 / Layer 2).

Each evidence base_root is routed to a real data source based on its agent_type
and task text:
  - financial  -> Gemma-orchestrated company intelligence via yfinance: a local
                  Gemma model resolves the query to specific public companies,
                  then yfinance supplies qualitative, non-price company data
                  (profile, ESG, governance risk, leadership, ownership, analyst
                  sentiment, company news). No price time-series is produced.
  - web_search -> Serper (if key) or DuckDuckGo; Wikipedia for structured background.

Every source is wrapped so a network/library failure degrades gracefully rather
than breaking the flow. All blocking library calls run in a worker thread to keep
the event loop responsive.
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field

from config import get_settings
from state import DagNode, EvidenceItem, TimeSeries

logger = logging.getLogger("singularity.evidence")

@dataclass
class EvidenceResult:
    items: list[EvidenceItem] = field(default_factory=list)
    series: list[TimeSeries] = field(default_factory=list)
    confidence: float = 0.6
    metadata: dict = field(default_factory=dict)


async def collect(
    node: DagNode, query: str, web_sources_enabled: bool = True
) -> EvidenceResult:
    result = await _route(node, query, web_sources_enabled)

    # Annotate every item with a lexicon polarity so the evidence feed can color
    # rows by sentiment. Sub-sources may set it explicitly; otherwise derive it.
    for item in result.items:
        if item.sentiment is None:
            item.sentiment = _score_sentiment(f"{item.title} {item.detail}")
    return result


async def _route(node: DagNode, query: str, web_sources_enabled: bool) -> EvidenceResult:
    """Select evidence sources. Routes through the LangChain tool layer when
    enabled (multi-source aggregation incl. arXiv/Parallel), else uses the
    hand-rolled path. Any failure in the tool layer falls back transparently."""
    if get_settings().langchain_enabled:
        try:
            from tools.registry import LANGCHAIN_AVAILABLE, collect_via_tools

            if LANGCHAIN_AVAILABLE:
                return await collect_via_tools(node, query, web_sources_enabled)
            logger.info("langchain_enabled but langchain not installed; using built-in evidence")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Tool-layer evidence failed, falling back: %s", exc)

    if node.agent_type == "financial":
        return await _financial(node, query, web_sources_enabled)
    return await _web(node, query, web_sources_enabled)


# --- company intelligence (Gemma-orchestrated yfinance) ----------------------
_TICKER_RE = re.compile(r"^[A-Za-z0-9.\-^]{1,15}$")


async def _financial(
    node: DagNode, query: str, web_sources_enabled: bool = True
) -> EvidenceResult:
    """Resolve relevant public companies via Gemma, then fetch non-price intel."""
    return await _company_intel(query, web_sources_enabled)


async def _company_intel(query: str, web_sources_enabled: bool = True) -> EvidenceResult:
    """Gemma resolves companies -> yfinance qualitative intel (no price series)."""
    if not web_sources_enabled:
        return EvidenceResult(items=[], series=[], confidence=0.0)

    companies = await resolve_companies(query)
    if not companies:
        return EvidenceResult(items=[], series=[], confidence=0.0)

    results = await asyncio.gather(
        *(
            asyncio.to_thread(_yfinance_company_blocking, c["ticker"], c["name"])
            for c in companies
        ),
        return_exceptions=True,
    )
    merged: list[EvidenceItem] = []
    for c, res in zip(companies, results):
        if isinstance(res, Exception):
            logger.warning("yfinance company intel failed (%s): %s", c.get("ticker"), res)
            continue
        merged.extend(res)

    if not merged:
        return EvidenceResult(items=[], series=[], confidence=0.0)
    return EvidenceResult(items=merged[: _MAX_EVIDENCE_ITEMS], series=[], confidence=0.8)


_MAX_EVIDENCE_ITEMS = 8


async def resolve_companies(query: str) -> list[dict]:
    """Use local Gemma to map the query to relevant public companies + tickers.

    Returns a validated, capped list of ``{"name", "ticker"}`` dicts; ``[]`` on any
    failure (so the company-intel node simply contributes nothing).
    """
    from llm.ollama_client import get_ollama
    from prompts import COMPANY_RESOLVER_SYSTEM, COMPANY_RESOLVER_USER

    settings = get_settings()
    max_companies = max(1, settings.company_intel_max_companies)
    try:
        data = await get_ollama().generate_json(
            COMPANY_RESOLVER_SYSTEM.format(max_companies=max_companies),
            COMPANY_RESOLVER_USER.format(query=query),
            temperature=0.2,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Company resolver (Gemma) failed: %s", exc)
        return []

    raw = data.get("companies") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        return []
    out: list[dict] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        ticker = str(item.get("ticker") or "").strip()
        if not name or not ticker or not _TICKER_RE.match(ticker):
            continue
        key = ticker.upper()
        if key in seen:
            continue
        seen.add(key)
        out.append({"name": name, "ticker": ticker})
        if len(out) >= max_companies:
            break
    return out


def _yfinance_company_blocking(ticker: str, name: str) -> list[EvidenceItem]:
    """Pull curated, non-price company intelligence for ``ticker`` as EvidenceItems.

    Every section is independently guarded so a missing/flaky field is skipped
    rather than aborting the whole company. Capped at the configured per-company
    item budget. Reputational signals (ESG, governance risk, analyst stance) carry
    an explicit sentiment; the rest are scored by the lexicon backfill in collect().
    """
    import yfinance as yf

    cap = max(1, get_settings().company_intel_max_items_per_company)
    tk = yf.Ticker(ticker)
    items: list[EvidenceItem] = []

    try:
        info = tk.info or {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("yfinance .info failed (%s): %s", ticker, exc)
        info = {}

    label = info.get("longName") or name or ticker

    # --- Profile ---
    try:
        sector = info.get("sector")
        industry = info.get("industry")
        summary = (info.get("longBusinessSummary") or "").strip()
        if sector or industry or summary:
            descriptor = " / ".join(p for p in (sector, industry) if p)
            detail = summary[:300] if summary else (descriptor or "Company profile.")
            items.append(
                EvidenceItem(
                    source="yFinance",
                    title=f"{label} - profile" + (f" ({descriptor})" if descriptor else ""),
                    detail=detail,
                    url=info.get("website"),
                )
            )
    except Exception as exc:  # noqa: BLE001
        logger.debug("profile section skipped (%s): %s", ticker, exc)

    # --- ESG / sustainability ---
    try:
        esg = tk.sustainability
        if esg is not None and not esg.empty:
            def _esg(key):
                return esg.loc[key].iloc[0] if key in esg.index else None

            total = _esg("totalEsg")
            env = _esg("environmentScore")
            soc = _esg("socialScore")
            gov = _esg("governanceScore")
            contro = _esg("highestControversy")
            if total is not None or contro is not None:
                parts = []
                if total is not None:
                    parts.append(f"Total ESG {float(total):.1f}")
                if env is not None:
                    parts.append(f"E {float(env):.1f}")
                if soc is not None:
                    parts.append(f"S {float(soc):.1f}")
                if gov is not None:
                    parts.append(f"G {float(gov):.1f}")
                if contro is not None:
                    parts.append(f"highest controversy {float(contro):.0f}/5")
                # Higher ESG-risk / controversy reads as more negative reputationally.
                sentiment = None
                if contro is not None:
                    sentiment = round(-min(1.0, float(contro) / 5.0), 3)
                elif total is not None:
                    sentiment = round(-min(1.0, float(total) / 40.0), 3)
                items.append(
                    EvidenceItem(
                        source="yFinance",
                        title=f"{label} - ESG & controversy",
                        detail="; ".join(parts) + ".",
                        sentiment=sentiment,
                    )
                )
    except Exception as exc:  # noqa: BLE001
        logger.debug("ESG section skipped (%s): %s", ticker, exc)

    # --- Governance risk ---
    try:
        overall = info.get("overallRisk")
        board = info.get("boardRisk")
        audit = info.get("auditRisk")
        rights = info.get("shareHolderRightsRisk")
        risk_vals = [v for v in (overall, board, audit, rights) if isinstance(v, (int, float))]
        if risk_vals:
            detail = (
                f"Governance risk (1=low,10=high): overall {overall}, board {board}, "
                f"audit {audit}, shareholder-rights {rights}."
            )
            # Risk scale 1..10 -> sentiment in roughly [-1, +0.8] (5.5 ~ neutral).
            mean_risk = sum(risk_vals) / len(risk_vals)
            sentiment = round(max(-1.0, min(1.0, (5.5 - mean_risk) / 4.5)), 3)
            items.append(
                EvidenceItem(
                    source="yFinance",
                    title=f"{label} - governance risk",
                    detail=detail,
                    sentiment=sentiment,
                )
            )
    except Exception as exc:  # noqa: BLE001
        logger.debug("governance section skipped (%s): %s", ticker, exc)

    # --- Leadership ---
    try:
        officers = info.get("companyOfficers") or []
        named = [
            f"{o.get('name')} ({o.get('title')})"
            for o in officers[:4]
            if o.get("name") and o.get("title")
        ]
        if named:
            items.append(
                EvidenceItem(
                    source="yFinance",
                    title=f"{label} - leadership",
                    detail="; ".join(named) + ".",
                )
            )
    except Exception as exc:  # noqa: BLE001
        logger.debug("leadership section skipped (%s): %s", ticker, exc)

    # --- Ownership ---
    try:
        held_inst = info.get("heldPercentInstitutions")
        held_ins = info.get("heldPercentInsiders")
        if isinstance(held_inst, (int, float)) or isinstance(held_ins, (int, float)):
            parts = []
            if isinstance(held_inst, (int, float)):
                parts.append(f"institutions {held_inst * 100:.1f}%")
            if isinstance(held_ins, (int, float)):
                parts.append(f"insiders {held_ins * 100:.1f}%")
            items.append(
                EvidenceItem(
                    source="yFinance",
                    title=f"{label} - ownership",
                    detail="Held by " + ", ".join(parts) + ".",
                )
            )
    except Exception as exc:  # noqa: BLE001
        logger.debug("ownership section skipped (%s): %s", ticker, exc)

    # --- Analyst sentiment ---
    try:
        rec_key = info.get("recommendationKey")
        rec_mean = info.get("recommendationMean")
        n_analysts = info.get("numberOfAnalystOpinions")
        if rec_key or isinstance(rec_mean, (int, float)):
            detail_parts = []
            if rec_key:
                detail_parts.append(f"consensus '{rec_key}'")
            if isinstance(rec_mean, (int, float)):
                detail_parts.append(f"mean {rec_mean:.2f} (1=buy,5=sell)")
            if isinstance(n_analysts, (int, float)):
                detail_parts.append(f"{int(n_analysts)} analysts")
            # recommendationMean 1..5 -> sentiment +1..-1 (3 ~ neutral hold).
            sentiment = None
            if isinstance(rec_mean, (int, float)):
                sentiment = round(max(-1.0, min(1.0, (3.0 - rec_mean) / 2.0)), 3)
            items.append(
                EvidenceItem(
                    source="yFinance",
                    title=f"{label} - analyst sentiment",
                    detail=", ".join(detail_parts) + ".",
                    sentiment=sentiment,
                )
            )
    except Exception as exc:  # noqa: BLE001
        logger.debug("analyst section skipped (%s): %s", ticker, exc)

    # --- Company news ---
    try:
        for article in (tk.news or [])[:2]:
            content = article.get("content", {}) if isinstance(article, dict) else {}
            title = (content.get("title") or "").strip()
            if not title:
                continue
            url = (content.get("canonicalUrl") or {}).get("url") if isinstance(
                content.get("canonicalUrl"), dict
            ) else None
            items.append(
                EvidenceItem(
                    source="yFinance",
                    title=f"{label} - news",
                    detail=title[:260],
                    url=url,
                )
            )
    except Exception as exc:  # noqa: BLE001
        logger.debug("news section skipped (%s): %s", ticker, exc)

    return items[:cap]


# --- web search --------------------------------------------------------------
async def _web(
    node: DagNode, query: str, web_sources_enabled: bool = True
) -> EvidenceResult:
    if not web_sources_enabled:
        return EvidenceResult(
            items=[EvidenceItem(source="Offline", title=node.task,
                                detail="Web sources disabled; proceeding on priors.")],
            confidence=0.3,
        )
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


# --- sentiment scoring -------------------------------------------------------
_POSITIVE_TERMS = frozenset({
    "growth", "grow", "gain", "rise", "rising", "surge", "positive", "strong",
    "boost", "optimism", "optimistic", "expand", "expansion", "accelerate",
    "accelerating", "opportunity", "success", "improve", "improving", "bullish",
    "record", "adopt", "adoption", "increase", "increasing", "recovery",
    "robust", "outperform", "momentum", "upgrade", "profit", "beat", "rally",
    "demand", "breakthrough", "upside",
})
_NEGATIVE_TERMS = frozenset({
    "decline", "declining", "loss", "fall", "falling", "drop", "negative",
    "weak", "weakness", "risk", "uncertain", "uncertainty", "concern", "fear",
    "slow", "slowdown", "recession", "crisis", "volatile", "volatility",
    "downturn", "bearish", "threat", "decrease", "layoff", "cut", "miss",
    "plunge", "fail", "failure", "warn", "warning", "drag", "headwind",
    "shortage", "downside", "barrier",
})


def _score_sentiment(text: str) -> float:
    """Lightweight lexicon polarity in [-1, 1] over free-form evidence text.

    Deterministic and dependency-free: counts positive vs negative term hits
    and normalizes by total hits. Neutral text (no hits) scores 0.0.
    """
    tokens = [t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split() if t]
    if not tokens:
        return 0.0
    pos = sum(1 for t in tokens if t in _POSITIVE_TERMS)
    neg = sum(1 for t in tokens if t in _NEGATIVE_TERMS)
    total = pos + neg
    if total == 0:
        return 0.0
    return max(-1.0, min(1.0, (pos - neg) / total))
