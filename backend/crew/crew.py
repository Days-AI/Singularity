"""Assemble and run the CrewAI synthesis crew, returning the report findings.

The crew kickoff is synchronous (CrewAI/litellm), so it runs in a worker thread.
The final editor task emits JSON matching the polish schema consumed by
``report.generate._to_sections``. Any failure raises so the caller can fall back
to the deterministic two-stage report builder.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re

from config import get_settings
from crew import personas
from state import SingularityState

logger = logging.getLogger("singularity.crew")

# Opt out of CrewAI telemetry / OTEL network calls in offline environments.
os.environ.setdefault("CREWAI_TELEMETRY_OPT_OUT", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)
_REQUIRED_KEYS = {"executive_summary", "key_findings", "strategic_implications", "risk_flags"}

_crewai_ok: bool | None = None


def crew_available() -> bool:
    """True only when CREWAI_ENABLED and crewai is importable."""
    global _crewai_ok
    if not get_settings().crewai_enabled:
        return False
    if _crewai_ok is None:
        try:
            import crewai  # noqa: F401

            _crewai_ok = True
        except Exception as exc:  # noqa: BLE001
            logger.warning("crewai unavailable: %s", exc)
            _crewai_ok = False
    return _crewai_ok


def _evidence_brief(data: dict) -> str:
    lines: list[str] = []
    wi = data.get("web_intelligence") or {}
    if wi.get("coverage_note"):
        lines.append(f"Coverage: {wi['coverage_note']}")
    for b in wi.get("source_breakdown", []):
        ms = b.get("mean_sentiment")
        sent = f", sentiment {ms:+.2f}" if ms is not None else ""
        lines.append(f"- Source **{b['source']}**: {b['count']} items{sent}")
    lines.append("")
    lines.append("Ranked findings (cite as [Source: Title]):")
    for e in data.get("evidence_highlights", [])[:12]:
        row = f"- [{e.get('source')}] {e.get('title')}: {e.get('detail')}"
        lines.append(row)
    for e in (wi.get("ranked_findings") or [])[:12]:
        if e.get("url"):
            lines.append(f"  URL: {e['url']}")
        if e.get("sentiment") is not None:
            lines.append(f"  Sentiment: {e['sentiment']:+.2f}")
    align = wi.get("simulation_alignment")
    if align:
        lines.append(
            f"Simulation alignment: {align['label']} "
            f"(evidence {align['evidence_sentiment']:+.2f} vs sim {align['simulation_sentiment']:+.2f})"
        )
    fc = data.get("forecast")
    if fc:
        lines.append(
            f"Forecast: {fc.get('model')} projects {fc.get('trend')} {fc.get('metric')} "
            f"({fc.get('pct_change')}% over {fc.get('horizon_days')}d, MASE {fc.get('mase_score')})."
        )
    for d in data.get("causal_drivers", [])[:5]:
        lines.append(f"Causal: {d['cause']} -> {d['effect']} (p={d['p_value']}, weight={d['weight']}).")
    if data.get("mean_sentiment") is not None:
        lines.append(f"Mean population sentiment: {data['mean_sentiment']:+.2f}.")
    for q in data.get("focus_questions", []):
        lines.append(f"Focus question: {q}")
    return "\n".join(lines) if lines else "No structured evidence available."


def _parse_findings(text: str) -> dict:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_BLOCK.search(text or "")
        if not match:
            raise ValueError("crew output contained no JSON object")
        obj = json.loads(match.group(0))
    if not isinstance(obj, dict) or not (_REQUIRED_KEYS & set(obj)):
        raise ValueError("crew JSON missing required report keys")
    return obj


def _kickoff_blocking(state: SingularityState, data: dict, persona_brief: str) -> str:
    from crewai import Crew, Process

    from crew.agents import build_agents, build_generation_llm, build_polish_llm, build_tasks

    tools = []
    try:
        from tools.registry import build_langchain_tools

        tools = build_langchain_tools(state.web_sources_enabled)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Skipping crew tools: %s", exc)

    agents = build_agents(build_generation_llm(), build_polish_llm(), tools=tools)
    tasks = build_tasks(
        agents,
        evidence_brief=_evidence_brief(data),
        persona_brief=persona_brief,
        rag_context=str(state.metrics.get("rag_context", "")),
        query=state.query,
    )
    crew = Crew(agents=list(agents.values()), tasks=tasks,
                process=Process.sequential, verbose=False)
    result = crew.kickoff()
    # CrewOutput stringifies to the final task output across crewai versions.
    return str(getattr(result, "raw", result))


async def synthesize(state: SingularityState, data: dict) -> dict:
    """Run the crew and return the polished findings dict. Raises on failure."""
    if not crew_available():
        raise RuntimeError("crew not available")
    persona_brief = personas.render_persona_context(personas.build_persona_context(state))
    raw = await asyncio.to_thread(_kickoff_blocking, state, data, persona_brief)
    return _parse_findings(raw)
