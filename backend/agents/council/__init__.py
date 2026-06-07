"""Specialist Agent Council — PR, Brand, Marketing, Consumer Psychology."""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from agents.council.prompts import SPECIALIST_PROMPTS
from agents.council.run import run_specialist
from config import get_settings
from llm.openrouter_client import get_openrouter
from state import (
    CouncilOpinionPayload,
    CouncilReadyPayload,
    SingularityState,
    SpecialistId,
)

logger = logging.getLogger("singularity.council")

SPECIALIST_IDS: list[SpecialistId] = ["pr", "brand", "marketing", "consumer_psychology"]


@dataclass
class CouncilResult:
    opinions: list[CouncilOpinionPayload] = field(default_factory=list)
    synthesis: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)


def _build_brief(state: SingularityState) -> str:
    lines = [f"Campaign brief / query: {state.query}"]
    if state.evidence:
        lines.append("Evidence highlights:")
        for e in state.evidence[:6]:
            lines.append(f"- [{e.source}] {e.title}")
    delib = state.metrics.get("deliberation", {})
    if delib:
        lines.append(
            f"Population: agreement {delib.get('agreement_rate', 0):.0%}, "
            f"polarization {delib.get('polarization_index', 0):.2f}"
        )
        clusters = delib.get("narrative_clusters", [])[:4]
        if clusters:
            lines.append("Narrative clusters:")
            for c in clusters:
                lines.append(f"  - {c.get('label')}: n={c.get('size')}, sent={c.get('sentiment')}")
    social = state.metrics.get("social_simulation", {})
    if social.get("final_narratives"):
        lines.append("Post-social narratives:")
        for n in social["final_narratives"][:5]:
            lines.append(f"  - {n.get('label')}: {n.get('adoption_pct')}% adoption")
    if state.persona_opinions:
        sample = _stratified_sample(state.persona_opinions, 12)
        lines.append("Sample agent voices:")
        for op in sample:
            comment = (op.comment or op.behavioral_intent)[:120]
            lines.append(f"  - [{op.cluster_label}] {comment}")
    return "\n".join(lines)


def _stratified_sample(opinions: list, n: int) -> list:
    by_cluster: dict[str, list] = {}
    for op in opinions:
        by_cluster.setdefault(op.cluster_label, []).append(op)
    out: list = []
    per = max(1, n // max(len(by_cluster), 1))
    for cluster_ops in by_cluster.values():
        out.extend(cluster_ops[:per])
    return out[:n]


async def _polish_synthesis(brief: str, opinions: list[CouncilOpinionPayload]) -> str:
    settings = get_settings()
    if not settings.council_polish_openrouter:
        return _template_synthesis(opinions)
    client = get_openrouter()
    if not (settings.council_polish_openrouter and client.api_available):
        return _template_synthesis(opinions)
    parts = "\n".join(
        f"{o.role}: {o.recommendation} (risks: {', '.join(o.risks[:2])})"
        for o in opinions
    )
    try:
        data = await client.chat_json(
            "You synthesize specialist council recommendations into one executive paragraph.",
            f"Brief:\n{brief}\n\nCouncil opinions:\n{parts}\n\n"
            'Return JSON: {"synthesis": "<2-3 sentence unified recommendation>"}',
            temperature=0.4,
        )
        return str(data.get("synthesis", ""))[:800] or _template_synthesis(opinions)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Council OpenRouter polish failed: %s", exc)
        return _template_synthesis(opinions)


def _template_synthesis(opinions: list[CouncilOpinionPayload]) -> str:
    if not opinions:
        return "Council did not produce recommendations."
    lead = opinions[0].recommendation
    others = sum(1 for o in opinions[1:] if o.confidence > 0.5)
    return (
        f"{lead} "
        f"Across {len(opinions)} specialists, {others} align on proceeding with measured rollout."
    )[:600]


async def run(
    state: SingularityState,
    emit_opinion: Callable[[CouncilOpinionPayload], Awaitable[None]] | None = None,
) -> CouncilResult:
    brief = _build_brief(state)
    settings = get_settings()
    sem = asyncio.Semaphore(max(1, settings.ollama_concurrency))

    async def one(spec_id: SpecialistId) -> CouncilOpinionPayload:
        async with sem:
            return await run_specialist(spec_id, brief, state.query)

    results = await asyncio.gather(*(one(sid) for sid in SPECIALIST_IDS))
    opinions: list[CouncilOpinionPayload] = []
    for op in results:
        opinions.append(op)
        if emit_opinion:
            await emit_opinion(op)

    synthesis = await _polish_synthesis(brief, opinions)
    metrics = {
        "opinions": [o.model_dump() for o in opinions],
        "synthesis": synthesis,
    }
    return CouncilResult(opinions=opinions, synthesis=synthesis, metrics=metrics)


def to_ready_payload(result: CouncilResult) -> CouncilReadyPayload:
    return CouncilReadyPayload(opinions=result.opinions, synthesis=result.synthesis)
