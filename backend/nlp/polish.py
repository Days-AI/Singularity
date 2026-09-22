"""Persona comment polish via shared naturalize layer."""
from __future__ import annotations

import logging

from config import get_settings
from nlp.cognitive_draft import CognitiveDraft
from nlp.naturalize import naturalize_draft

logger = logging.getLogger("singularity.nlp.polish")


async def polish_persona_comment(
    comment: str,
    *,
    ocean_summary: str,
    topic: str,
    voice_register: str,
    sentiment: float,
    intent: str,
    agent_id: str = "llm_sample",
    entropy: float = 0.5,
    confidence: float = 0.5,
) -> tuple[str, str]:
    """Return (polished_comment, source) where source is llm_polished or llm."""
    settings = get_settings()
    if not comment or len(comment) < 12:
        return comment, "llm"

    if settings.skip_cognitive_nlg_extras or (
        not settings.persona_polish_openrouter and settings.latency_mode_normalized != "quality"
    ):
        return comment, "llm"

    draft = CognitiveDraft(
        agent_id=agent_id,
        topic=topic or "this",
        domain_label="general",
        stimulus_snippet="",
        sentiment=sentiment,
        confidence=confidence,
        behavioral_intent=intent,
        winning_voice="",
        key_concerns=[],
        emotions={},
        deliberation_summary=[],
        ocean_summary=ocean_summary,
        voice_register=voice_register,
        entropy=entropy,
    )
    polished, src = await naturalize_draft(draft, raw_comment=comment)
    if src in ("llm_polished", "naturalized"):
        return polished, "llm_polished"
    return comment, "llm"
