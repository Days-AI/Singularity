"""Voice selection utilities for persona NLG (no preset phrase banks)."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

from state import FacetScore, OceanScores

if TYPE_CHECKING:
    from agents.cognitive.types import AgentCognitiveOutput
    from nlp.topic_context import TopicContext

_STOPWORDS = {
    "analyze", "predict", "forecast", "market", "sentiment", "consumer",
    "quarter", "next", "about", "with", "from", "that", "this", "what", "when",
    "where", "which", "the", "and", "for", "behavioral", "drivers", "trends",
}


def sentiment_band(sentiment: float) -> str:
    if sentiment >= 0.15:
        return "positive"
    if sentiment <= -0.15:
        return "negative"
    return "neutral"


def topic_keyword(query: str) -> str:
    words = [w.strip(".,?!:;\"'") for w in (query or "").split()]
    words = [w for w in words if len(w) > 3 and w.lower() not in _STOPWORDS]
    return " ".join(words[:3]) if words else "this"


def voice_register(ocean: OceanScores, p_idx: int) -> str:
    """Pick a stylistic register from the full OCEAN profile."""
    candidates: list[str] = []
    if ocean.O >= 60 or ocean.C >= 65:
        candidates.append("analyst")
    if ocean.N >= 60 or ocean.A <= 40:
        candidates.append("skeptic")
    if ocean.E >= 62 and ocean.N <= 52:
        candidates.append("hype")
    if ocean.C >= 58:
        candidates.append("pragmatist")
    if ocean.A >= 62:
        candidates.append("empath")
    if ocean.E >= 58 and ocean.A <= 45:
        candidates.append("blunt")
    candidates.append("casual")
    return random.Random(p_idx * 7 + 13).choice(candidates)


@dataclass
class VoiceProfile:
    register: str
    formality: float = 0.5
    hedge_rate: float = 0.3


def select_voice(
    ocean: OceanScores,
    p_idx: int,
    entropy: float = 0.0,
    rng: random.Random | None = None,
) -> VoiceProfile:
    r = rng or random.Random(p_idx * 7 + 13)
    register = voice_register(ocean, p_idx)
    if entropy > 0.55 and r.random() < entropy * 0.35:
        alt = ["skeptic", "empath", "casual", "analyst"]
        register = r.choice([register] + alt)
    formality = 0.7 if register in ("analyst", "pragmatist") else 0.35
    hedge = 0.15 + entropy * 0.35
    return VoiceProfile(register=register, formality=formality, hedge_rate=hedge)


def _draft_helpers():
    from nlp.cognitive_draft import build_draft_from_persona_fields, draft_fallback_line
    return build_draft_from_persona_fields, draft_fallback_line


def persona_comment(
    p_idx: int,
    ocean: OceanScores,
    top_facets: list[FacetScore],
    sentiment: float,
    cluster_label: str,
    topic: str,
    arch_intent: str = "",
    arch_emotion: str = "",
    entropy: float = 0.5,
    source_sentiment: float | None = None,
) -> str:
    """Sync compositional fallback for tests and non-async callers."""
    _ = source_sentiment
    build_draft_from_persona_fields, draft_fallback_line = _draft_helpers()
    register = voice_register(ocean, p_idx)
    draft = build_draft_from_persona_fields(
        agent_id=f"p_{p_idx:04d}",
        ocean=ocean,
        sentiment=sentiment,
        behavioral_intent=arch_intent or "evaluating options",
        emotional_state=arch_emotion or "measured",
        key_concerns=[],
        action_likelihood=0.5,
        cluster_label=cluster_label,
        top_facets=top_facets,
        topic=topic,
        voice_register=register,
        entropy=entropy,
    )
    return draft_fallback_line(draft)


def compose_programmatic_fallback(
    output: AgentCognitiveOutput,
    topic_ctx: TopicContext | None,
    stimulus: str,
    rng: random.Random,
) -> str:
    """Compositional fallback when async naturalize is unavailable."""
    from nlp.cognitive_draft import build_cognitive_draft, draft_fallback_line

    p_idx = int(output.state.agent_id.split("_")[-1]) if "_" in output.state.agent_id else 0
    voice = select_voice(output.state.ocean, p_idx, output.state.total_entropy, rng)
    draft = build_cognitive_draft(
        output, topic_ctx, stimulus, voice_register=voice.register
    )
    return draft_fallback_line(draft)


async def compose_programmatic(
    output: AgentCognitiveOutput,
    topic_ctx: TopicContext | None,
    rng: random.Random,
    *,
    stimulus: str = "",
) -> tuple[str, str]:
    """Build cognitive draft and naturalize via local Gemma."""
    from nlp.cognitive_draft import build_cognitive_draft
    from nlp.naturalize import naturalize_draft

    p_idx = int(output.state.agent_id.split("_")[-1]) if "_" in output.state.agent_id else 0
    voice = select_voice(output.state.ocean, p_idx, output.state.total_entropy, rng)
    draft = build_cognitive_draft(
        output, topic_ctx, stimulus, voice_register=voice.register
    )
    return await naturalize_draft(draft)


def naturalize_spacy(text: str, voice: VoiceProfile, topic: str = "") -> str:
    """Light spaCy sentence cleanup."""
    try:
        from nlp.topic_context import _load_spacy

        nlp = _load_spacy()
        if not nlp or len(text) < 12:
            return text
        doc = nlp(text)
        sents = [s.text.strip() for s in doc.sents if s.text.strip()]
        out = " ".join(sents) if sents else text
        return out[:240]
    except Exception:
        return text[:240]
