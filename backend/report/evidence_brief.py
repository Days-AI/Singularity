"""Structured web-search / source intelligence for report synthesis.

Turns ``SingularityState.evidence`` into ranked findings, source breakdown,
and McKinsey-style markdown for the External Intelligence section.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from state import EvidenceItem, SingularityState

# Sources treated as internal simulation (not external web).
_INTERNAL_SOURCES = frozenset({"Simulation", "PsychometricEngine", "IPIP-300"})


def _mean_sentiment(items: list[EvidenceItem]) -> float | None:
    vals = [e.sentiment for e in items if e.sentiment is not None]
    if not vals:
        return None
    return round(float(np.mean(vals)), 3)


def _finding_dict(e: EvidenceItem) -> dict:
    row: dict = {
        "source": e.source,
        "title": e.title,
        "detail": e.detail,
    }
    if e.url:
        row["url"] = e.url
    if e.sentiment is not None:
        row["sentiment"] = round(float(e.sentiment), 3)
    if e.value is not None:
        row["value"] = e.value
    if e.unit:
        row["unit"] = e.unit
    return row


def _rank_findings(items: list[EvidenceItem], max_items: int) -> list[dict]:
    """Rank by absolute sentiment strength, then title uniqueness."""
    seen_titles: set[str] = set()
    unique: list[EvidenceItem] = []
    for e in items:
        key = e.title.strip().lower()
        if key in seen_titles:
            continue
        seen_titles.add(key)
        unique.append(e)

    def sort_key(e: EvidenceItem) -> tuple[float, str]:
        sent = abs(e.sentiment) if e.sentiment is not None else 0.0
        return (-sent, e.title)

    ranked = sorted(unique, key=sort_key)[:max_items]
    return [_finding_dict(e) for e in ranked]


def build_web_intelligence(state: SingularityState, max_items: int = 12) -> dict:
    """Build report-ready web intelligence from collected evidence."""
    evidence = list(state.evidence)
    external = [e for e in evidence if e.source not in _INTERNAL_SOURCES]
    internal = [e for e in evidence if e.source in _INTERNAL_SOURCES]

    # Prefer external web/financial items; fall back to all evidence.
    pool = external if external else evidence

    by_source: dict[str, list[EvidenceItem]] = defaultdict(list)
    for e in pool:
        by_source[e.source].append(e)

    source_breakdown: list[dict] = []
    for source, items in sorted(by_source.items(), key=lambda x: -len(x[1])):
        ms = _mean_sentiment(items)
        source_breakdown.append({
            "source": source,
            "count": len(items),
            "mean_sentiment": ms,
            "sample_titles": [i.title for i in items[:3]],
        })

    ranked_findings = _rank_findings(pool, max_items)

    persona_sent = None
    if state.persona_responses:
        persona_sent = round(
            float(np.mean([r.sentiment_score for r in state.persona_responses])), 3
        )

    evidence_sent = _mean_sentiment(pool)
    simulation_alignment: dict | None = None
    if evidence_sent is not None and persona_sent is not None:
        delta = round(persona_sent - evidence_sent, 3)
        aligned = abs(delta) <= 0.15
        simulation_alignment = {
            "evidence_sentiment": evidence_sent,
            "simulation_sentiment": persona_sent,
            "delta": delta,
            "label": "aligned" if aligned else "divergent",
        }

    providers = sorted(by_source.keys())
    if not providers:
        coverage_note = "No live web evidence collected for this run."
    elif external:
        coverage_note = f"External intelligence from {len(providers)} source(s): {', '.join(providers)}."
    else:
        coverage_note = f"Internal simulation evidence only ({', '.join(providers)})."

    return {
        "source_breakdown": source_breakdown,
        "ranked_findings": ranked_findings,
        "simulation_alignment": simulation_alignment,
        "coverage_note": coverage_note,
        "external_count": len(external),
        "internal_count": len(internal),
        "total_count": len(evidence),
    }


def _sentiment_label(sent: float | None) -> str:
    if sent is None:
        return "neutral"
    if sent >= 0.15:
        return "positive"
    if sent <= -0.15:
        return "negative"
    return "neutral"


def build_external_intelligence_md(
    intel: dict,
    query: str,
    llm_summary: str | None = None,
) -> str:
    """Deterministic McKinsey-style External Intelligence markdown."""
    lines: list[str] = []
    coverage = intel.get("coverage_note", "")
    lines.append(f"**Situation.** Analysis of *{query or 'the query'}*. {coverage}")

    if llm_summary and llm_summary.strip():
        lines.append("")
        lines.append(llm_summary.strip())

    findings = intel.get("ranked_findings", [])
    if findings:
        lines.append("")
        lines.append("**Key external signals**")
        for i, f in enumerate(findings, 1):
            src = f.get("source", "Unknown")
            title = f.get("title", "")
            detail = (f.get("detail") or "")[:220]
            sent = f.get("sentiment")
            sent_note = f" ({_sentiment_label(sent)})" if sent is not None else ""
            lines.append(f"{i}. **[{src}]** {title}{sent_note} — {detail}")

    breakdown = intel.get("source_breakdown", [])
    if breakdown:
        lines.append("")
        lines.append("**Source coverage**")
        for b in breakdown:
            ms = b.get("mean_sentiment")
            sent_str = f", mean sentiment {ms:+.2f}" if ms is not None else ""
            samples = b.get("sample_titles", [])
            sample_str = f" — e.g. {samples[0][:60]}" if samples else ""
            lines.append(
                f"- **{b['source']}** ({b['count']} item{'s' if b['count'] != 1 else ''}{sent_str}){sample_str}"
            )

    align = intel.get("simulation_alignment")
    if align:
        lines.append("")
        lines.append(
            f"**Alignment with simulation.** External evidence sentiment "
            f"({align['evidence_sentiment']:+.2f}) vs simulated population "
            f"({align['simulation_sentiment']:+.2f}) is **{align['label']}** "
            f"(delta {align['delta']:+.2f})."
        )

    urls = [f for f in findings if f.get("url")]
    if urls:
        lines.append("")
        lines.append("**Source appendix**")
        for f in urls:
            lines.append(f"- [{f.get('source')}] [{f.get('title', 'link')}]({f['url']})")

    if not findings and not breakdown:
        lines.append("")
        lines.append("_No external intelligence items were collected. Enable web sources and re-run._")

    return "\n".join(lines)


def findings_to_highlights(ranked_findings: list[dict]) -> list[dict]:
    """Compact alias for Crew backward compatibility."""
    return [
        {"source": f["source"], "title": f["title"], "detail": f["detail"]}
        for f in ranked_findings
    ]
