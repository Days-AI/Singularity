"""Report synthesis (spec sections 3 step 6-7).

Two-stage: local Gemma drafts evidence-grounded findings from the structured
simulation data, then the OpenRouter polishing layer (PT-03) elevates it to
executive-grade prose. Both stages degrade gracefully - if OpenRouter is
unconfigured the local model produces the structured JSON; if the LLM is down
entirely a deterministic template assembles the sections from the hard numbers.

Emits four sections matching the dashboard's ReportViewer:
  Executive Summary | Key Findings | Strategic Implications | Risk Flags
"""
from __future__ import annotations

import json
import logging

import numpy as np

from llm.ollama_client import get_ollama
from llm.openrouter_client import get_openrouter
from prompts import (
    FINDINGS_SYSTEM,
    FINDINGS_USER,
    POLISH_SYSTEM,
    POLISH_USER,
)
from state import ReportSectionPayload, SingularityState

logger = logging.getLogger("singularity.report")

_SECTIONS = ["Executive Summary", "Key Findings", "Strategic Implications", "Risk Flags"]


async def build(state: SingularityState) -> list[ReportSectionPayload]:
    data = _structured_data(state)
    raw = await _gemma_findings(data)
    polished = await _polish(raw, data)
    return _to_sections(polished)


def _structured_data(state: SingularityState) -> dict:
    om = state.ocean_mean
    fc = state.forecast
    forecast_summary = None
    if fc and fc.predictions:
        first, last = fc.predictions[0].value, fc.predictions[-1].value
        trend = "rising" if last > first else "declining" if last < first else "flat"
        forecast_summary = {
            "model": fc.model, "metric": fc.metric, "horizon_days": fc.horizon_days,
            "mase_score": fc.mase_score, "start_value": round(first, 2),
            "end_value": round(last, 2), "trend": trend,
            "pct_change": round((last - first) / first * 100, 1) if first else 0.0,
        }
    causal_edges = []
    if state.causal:
        node_label = {n.id: n.label for n in state.causal.nodes}
        for e in sorted(state.causal.edges, key=lambda x: -x.weight)[:5]:
            causal_edges.append({
                "cause": node_label.get(e.source, e.source),
                "effect": node_label.get(e.target, e.target),
                "p_value": e.p_value, "weight": e.weight, "lag": e.lag,
            })
    mean_sentiment = (
        round(float(np.mean([r.sentiment_score for r in state.persona_responses])), 3)
        if state.persona_responses else None
    )
    data = {
        "query": state.query,
        "evidence_highlights": [
            {"source": e.source, "title": e.title, "detail": e.detail}
            for e in state.evidence[:8]
        ],
        "population": state.metrics.get("personas", len(state.persona_responses)),
        "ocean_mean": om.model_dump() if om else None,
        "mean_sentiment": mean_sentiment,
        "forecast": forecast_summary,
        "causal_drivers": causal_edges,
    }
    # Optional user focus questions steer the synthesis toward specific angles.
    focus_questions = [
        q.strip() for q in state.metrics.get("focus_questions", []) if str(q).strip()
    ]
    if focus_questions:
        data["focus_questions"] = focus_questions
    return data


async def _gemma_findings(data: dict) -> str:
    try:
        return await get_ollama().generate(
            FINDINGS_SYSTEM, FINDINGS_USER.format(data=json.dumps(data, default=str)),
            temperature=0.5, max_tokens=700,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Gemma findings fallback: %s", exc)
        return _template_findings(data)


async def _polish(raw: str, data: dict) -> dict:
    payload = json.dumps(data, default=str)
    openrouter = get_openrouter()
    if openrouter.enabled:
        try:
            return await openrouter.chat_json(
                POLISH_SYSTEM, POLISH_USER.format(raw=raw, data=payload)
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("OpenRouter polish fallback to local: %s", exc)
    try:
        return await get_ollama().generate_json(
            POLISH_SYSTEM, POLISH_USER.format(raw=raw, data=payload), temperature=0.5
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Local polish fallback to template: %s", exc)
        return _template_polished(raw, data)


def _to_sections(polished: dict) -> list[ReportSectionPayload]:
    exec_summary = str(polished.get("executive_summary", "")).strip()
    findings = polished.get("key_findings", [])
    implications = str(polished.get("strategic_implications", "")).strip()
    risks = polished.get("risk_flags", [])

    findings_md = "\n".join(f"- {str(f).strip()}" for f in findings) if isinstance(findings, list) else str(findings)
    risks_md = "\n".join(f"- {str(r).strip()}" for r in risks) if isinstance(risks, list) else str(risks)

    contents = [exec_summary, findings_md, implications, risks_md]
    sections: list[ReportSectionPayload] = []
    for i, (title, content) in enumerate(zip(_SECTIONS, contents)):
        sections.append(
            ReportSectionPayload(
                index=i, total=len(_SECTIONS), section=title,
                content=content or "_No data available for this section._",
                status="final",
            )
        )
    return sections


# --- deterministic fallbacks -------------------------------------------------
def _template_findings(data: dict) -> str:
    fc = data.get("forecast") or {}
    sent = data.get("mean_sentiment")
    parts = [f"Analysis of '{data.get('query')}' across {data.get('population')} simulated personas."]
    if sent is not None:
        parts.append(f"Aggregate consumer sentiment is {sent:+.2f} on a -1..1 scale.")
    if fc:
        parts.append(
            f"The {fc.get('model')} model projects a {fc.get('trend')} {fc.get('metric')} "
            f"({fc.get('pct_change')}% over {fc.get('horizon_days')} days, MASE {fc.get('mase_score')})."
        )
    for d in data.get("causal_drivers", [])[:3]:
        parts.append(f"{d['cause']} Granger-causes {d['effect']} (p={d['p_value']}, weight={d['weight']}).")
    for q in data.get("focus_questions", []):
        parts.append(f"Focus question: {q}")
    return " ".join(parts)


def _template_polished(raw: str, data: dict) -> dict:
    fc = data.get("forecast") or {}
    sent = data.get("mean_sentiment")
    findings = []
    if sent is not None:
        findings.append(f"Population sentiment registers {sent:+.2f}, indicating "
                        f"{'net-positive' if sent > 0 else 'net-cautious'} disposition.")
    if fc:
        findings.append(f"{fc.get('model')} projects {fc.get('pct_change')}% movement in "
                        f"{fc.get('metric')} over {fc.get('horizon_days')} days (MASE {fc.get('mase_score')}).")
    for d in data.get("causal_drivers", [])[:4]:
        findings.append(f"{d['cause']} -> {d['effect']}: significant at p={d['p_value']} "
                        f"(excitation {d['weight']}, lag {d['lag']}).")
    if data.get("ocean_mean"):
        om = data["ocean_mean"]
        findings.append(f"Mean OCEAN profile O{om['O']:.0f}/C{om['C']:.0f}/E{om['E']:.0f}/"
                        f"A{om['A']:.0f}/N{om['N']:.0f} shapes messaging strategy.")
    return {
        "executive_summary": raw[:900],
        "key_findings": findings or ["Insufficient signal to extract findings."],
        "strategic_implications": (
            "Prioritize interventions on the highest-excitation causal drivers and align "
            "go-to-market messaging to the dominant personality segments. Monitor the "
            "forecast trajectory and re-run the simulation as fresh evidence arrives."
        ),
        "risk_flags": [
            "Forecast assumes evidence regime persists; structural breaks invalidate intervals.",
            "Persona expansion is statistically derived from archetypes, not full live sampling.",
        ],
    }
