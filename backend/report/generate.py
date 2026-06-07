"""Report synthesis (spec sections 3 step 6-7).

Two-stage: local Gemma drafts evidence-grounded findings from the structured
simulation data, then the OpenRouter polishing layer (PT-03) elevates it to
executive-grade prose. Both stages degrade gracefully - if OpenRouter is
unconfigured the local model produces the structured JSON; if the LLM is down
entirely a deterministic template assembles the sections from the hard numbers.

Emits seven consulting-style sections matching the dashboard ReportViewer:
  Executive Summary | External Intelligence & Sources | Key Findings |
  Strategic Implications | Risk Flags | Simulation Applications | Council Consensus
"""
from __future__ import annotations

import json
import logging

import numpy as np

from agents.causal import compute_outcome_probability
from config import get_settings
from llm.ollama_client import get_ollama
from llm.openrouter_client import get_openrouter
from prompts import (
    FINDINGS_SYSTEM,
    FINDINGS_USER,
    POLISH_SYSTEM,
    POLISH_USER,
)
from report.evidence_brief import (
    build_external_intelligence_md,
    build_web_intelligence,
    findings_to_highlights,
)
from report.use_cases import build_application_playbooks_markdown, merge_llm_playbooks
from state import ReportSectionPayload, SingularityState

logger = logging.getLogger("singularity.report")

_SECTIONS = [
    "Executive Summary",
    "External Intelligence & Sources",
    "Key Findings",
    "Strategic Implications",
    "Risk Flags",
    "Simulation Applications",
    "Council Consensus",
]


async def build(state: SingularityState) -> list[ReportSectionPayload]:
    data = _structured_data(state)
    polished = await _crew_findings(state, data)
    if polished is None:
        raw = await _gemma_findings(data)
        polished = await _polish(raw, data)
    return _to_sections(polished, data)


async def _crew_findings(state: SingularityState, data: dict) -> dict | None:
    """Produce findings via the persona-injected CrewAI layer when enabled.

    Returns None to signal the caller should use the default Gemma->polish path
    (crew disabled, unavailable, or failed).
    """
    if not get_settings().crewai_enabled:
        return None
    try:
        from crew import crew_available, synthesize

        if not crew_available():
            return None
        polished = await synthesize(state, data)
        logger.info("Report synthesized via CrewAI layer")
        return polished
    except Exception as exc:  # noqa: BLE001
        logger.warning("CrewAI synthesis failed, falling back to Gemma->polish: %s", exc)
        return None


def _structured_data(state: SingularityState) -> dict:
    settings = get_settings()
    web_intel = build_web_intelligence(state, max_items=settings.report_evidence_max_items)

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
        "outcome_probability": round(compute_outcome_probability(state), 1),
        "web_intelligence": web_intel,
        "evidence_highlights": findings_to_highlights(web_intel.get("ranked_findings", [])),
        "population": state.metrics.get("personas", len(state.persona_responses)),
        "ocean_mean": om.model_dump() if om else None,
        "mean_sentiment": mean_sentiment,
        "forecast": forecast_summary,
        "causal_drivers": causal_edges,
    }
    focus_questions = [
        q.strip() for q in state.metrics.get("focus_questions", []) if str(q).strip()
    ]
    if focus_questions:
        data["focus_questions"] = focus_questions
    rag_context = state.metrics.get("rag_context")
    if rag_context:
        data["rag_context"] = str(rag_context)
    deliberation = state.metrics.get("deliberation")
    if deliberation:
        data["deliberation"] = {
            "agreement_rate": deliberation.get("agreement_rate"),
            "polarization_index": deliberation.get("polarization_index"),
            "narrative_clusters": deliberation.get("narrative_clusters", [])[:5],
            "persona_archetypes": deliberation.get("persona_archetypes", [])[:3],
        }
    prediction_market = state.metrics.get("prediction_market")
    if prediction_market:
        data["prediction_market"] = {
            "overall_outcome": prediction_market.get("overall_outcome"),
            "confidence_interval": prediction_market.get("confidence_interval"),
            "forecasts": prediction_market.get("forecasts", [])[:5],
            "probability_distribution": prediction_market.get("probability_distribution"),
        }
    monte_carlo = state.metrics.get("monte_carlo")
    if monte_carlo:
        data["monte_carlo"] = {
            "outcome_percentiles": monte_carlo.get("outcome_percentiles"),
            "best_case": monte_carlo.get("best_case"),
            "worst_case": monte_carlo.get("worst_case"),
            "most_likely": monte_carlo.get("most_likely"),
            "black_swan": monte_carlo.get("black_swan"),
        }
    decision_engine = state.metrics.get("decision_engine")
    if decision_engine:
        data["decision_engine"] = decision_engine.get("options", [])[:5]
    swarm = state.metrics.get("swarm_optimization")
    if swarm:
        data["swarm_optimization"] = {
            "domain": swarm.get("domain"),
            "algorithm": swarm.get("algorithm"),
            "optimal_path": swarm.get("optimal_path"),
            "alternatives": swarm.get("alternatives", [])[:3],
            "convergence_iterations": swarm.get("convergence_iterations"),
        }
    social = state.metrics.get("social_simulation")
    if social:
        data["social_simulation"] = {
            "rounds_completed": social.get("rounds_completed"),
            "contagion_index": social.get("contagion_index"),
            "polarization_index": social.get("polarization_index"),
            "final_narratives": social.get("final_narratives", [])[:5],
        }
    council = state.metrics.get("council")
    if council:
        data["council"] = {
            "synthesis": council.get("synthesis"),
            "opinions": council.get("opinions", [])[:4],
        }
    consensus = state.metrics.get("consensus")
    if consensus:
        data["consensus"] = {
            "agreement_score": consensus.get("agreement_score"),
            "recommended_action": consensus.get("recommended_action"),
            "dissent": consensus.get("dissent"),
            "supporting_signals": consensus.get("supporting_signals", [])[:6],
            "council_alignment": consensus.get("council_alignment"),
            "ranked_actions": consensus.get("ranked_actions", [])[:5],
        }
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
            POLISH_SYSTEM, POLISH_USER.format(raw=raw, data=payload), temperature=0.5,
            max_tokens=get_settings().report_polish_max_tokens,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Local polish fallback to template: %s", exc)
        return _template_polished(raw, data)


def _to_sections(polished: dict, data: dict) -> list[ReportSectionPayload]:
    exec_summary = str(polished.get("executive_summary", "")).strip()
    ext_summary = str(polished.get("external_intelligence_summary", "")).strip()
    web_intel = data.get("web_intelligence") or {}
    external_md = build_external_intelligence_md(
        web_intel, str(data.get("query", "")), llm_summary=ext_summary or None
    )

    findings = polished.get("key_findings", [])
    implications = str(polished.get("strategic_implications", "")).strip()
    risks = polished.get("risk_flags", [])

    findings_md = "\n".join(f"- {str(f).strip()}" for f in findings) if isinstance(findings, list) else str(findings)
    risks_md = "\n".join(f"- {str(r).strip()}" for r in risks) if isinstance(risks, list) else str(risks)

    llm_playbooks = merge_llm_playbooks(polished)
    applications_md = build_application_playbooks_markdown(data, llm_playbooks)
    consensus_md = _consensus_section_md(data)

    contents = [
        exec_summary,
        external_md,
        findings_md,
        implications,
        risks_md,
        applications_md,
        consensus_md,
    ]
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


def _consensus_section_md(data: dict) -> str:
    consensus = data.get("consensus")
    council = data.get("council")
    if not consensus and not council:
        return "_Council consensus not available for this run._"
    lines: list[str] = []
    if consensus:
        score = consensus.get("agreement_score", 0)
        lines.append(f"**Agreement score:** {score:.0%}")
        rec = consensus.get("recommended_action", "")
        if rec:
            lines.append(f"\n**Recommended action:** {rec}")
        dissent = consensus.get("dissent", "")
        if dissent:
            lines.append(f"\n**Dissent:** {dissent}")
        signals = consensus.get("supporting_signals", [])
        if signals:
            lines.append("\n**Supporting signals:**")
            lines.extend(f"- {s}" for s in signals)
        ranked = consensus.get("ranked_actions", [])
        if ranked:
            lines.append("\n**Ranked actions:**")
            for i, action in enumerate(ranked, 1):
                lines.append(f"{i}. {action}")
    if council and council.get("opinions"):
        lines.append("\n**Specialist council:**")
        for op in council["opinions"]:
            role = op.get("role", op.get("specialist_id", "Specialist"))
            lines.append(f"- **{role}:** {op.get('recommendation', '')[:200]}")
        if council.get("synthesis"):
            lines.append(f"\n**Council synthesis:** {council['synthesis']}")
    return "\n".join(lines) if lines else "_No consensus data._"


def _web_finding_bullets(data: dict, limit: int = 4) -> list[str]:
    """Template bullets citing external sources."""
    bullets: list[str] = []
    for f in (data.get("web_intelligence") or {}).get("ranked_findings", [])[:limit]:
        src = f.get("source", "Source")
        title = f.get("title", "")
        detail = (f.get("detail") or "")[:100]
        bullets.append(f"**[{src}: {title}]** — {detail}")
    return bullets


# --- deterministic fallbacks -------------------------------------------------
def _template_findings(data: dict) -> str:
    fc = data.get("forecast") or {}
    sent = data.get("mean_sentiment")
    parts = [f"Analysis of '{data.get('query')}' across {data.get('population')} simulated personas."]
    wi = data.get("web_intelligence") or {}
    if wi.get("coverage_note"):
        parts.append(wi["coverage_note"])
    for bullet in _web_finding_bullets(data, 3):
        parts.append(bullet)
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
    findings = list(_web_finding_bullets(data, 4))
    if sent is not None:
        findings.append(
            f"**Population sentiment** registers {sent:+.2f}, indicating "
            f"{'net-positive' if sent > 0 else 'net-cautious'} disposition."
        )
    if fc:
        findings.append(
            f"**Forecast trajectory:** {fc.get('model')} projects {fc.get('pct_change')}% movement in "
            f"{fc.get('metric')} over {fc.get('horizon_days')} days (MASE {fc.get('mase_score')})."
        )
    for d in data.get("causal_drivers", [])[:4]:
        findings.append(
            f"**Causal driver:** {d['cause']} -> {d['effect']} significant at p={d['p_value']} "
            f"(excitation {d['weight']}, lag {d['lag']})."
        )
    if data.get("ocean_mean"):
        om = data["ocean_mean"]
        findings.append(
            f"**Behavioral profile:** Mean OCEAN O{om['O']:.0f}/C{om['C']:.0f}/E{om['E']:.0f}/"
            f"A{om['A']:.0f}/N{om['N']:.0f} shapes messaging strategy."
        )
    delib = data.get("deliberation")
    if delib:
        findings.append(
            f"**Deliberation:** agreement {delib.get('agreement_rate', 0):.0%}, "
            f"polarization {delib.get('polarization_index', 0):.2f}."
        )
    mc = data.get("monte_carlo")
    if mc and mc.get("outcome_percentiles"):
        p = mc["outcome_percentiles"]
        findings.append(
            f"**Monte Carlo:** median {p.get('p50', 50):.0f}/100 "
            f"(range {p.get('p5', 0):.0f}–{p.get('p95', 100):.0f})."
        )
    swarm = data.get("swarm_optimization")
    if swarm:
        findings.append(
            f"**Swarm optimization:** {swarm.get('algorithm', 'maco').upper()} "
            f"{swarm.get('domain', 'path')} pathway (score "
            f"{swarm.get('optimal_path', {}).get('score', 0):.2f})."
        )
    for opt in data.get("decision_engine", [])[:2]:
        findings.append(f"[{opt.get('type', 'action').upper()}] {opt.get('action', '')[:120]}")

    wi = data.get("web_intelligence") or {}
    ext_summary = wi.get("coverage_note", "")
    ranked = wi.get("ranked_findings", [])
    if ranked:
        top = ranked[0]
        ext_summary = (
            f"External intelligence from {top.get('source')} and "
            f"{len(wi.get('source_breakdown', []))} provider(s) "
            f"supports directional analysis. {wi.get('coverage_note', '')}"
        )

    return {
        "executive_summary": (
            f"The simulation estimates a **{data.get('outcome_probability', 50):.1f}/100** "
            f"headline outcome probability for \"{data.get('query', 'this query')}\" "
            f"across {data.get('population', 0)} personas. "
            + raw[:700]
        ),
        "external_intelligence_summary": ext_summary,
        "key_findings": findings or ["Insufficient signal to extract findings."],
        "strategic_implications": (
            "Prioritize interventions on the highest-excitation causal drivers and align "
            "audience messaging and creative strategy to the dominant personality segments. "
            "Cross-check external intelligence with simulation divergence before committing spend. "
            "Monitor the forecast trajectory and re-run the simulation as fresh evidence arrives."
        ),
        "risk_flags": [
            "Forecast assumes evidence regime persists; structural breaks invalidate intervals.",
            "Persona expansion is statistically derived from archetypes, not full live sampling.",
            "Monte Carlo scenarios include low-probability tail events; black swan paths are not base case.",
        ],
    }
