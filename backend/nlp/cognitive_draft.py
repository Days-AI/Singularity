"""Structured cognitive briefs for natural-language generation (no phrase banks)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from nlp.prompt_budget import clamp_text, compact_debate_lines
from state import FacetScore, OceanScores

_STOPWORDS = {
    "analyze", "predict", "forecast", "market", "sentiment", "consumer",
    "quarter", "next", "about", "with", "from", "that", "this", "what", "when",
    "where", "which", "the", "and", "for", "behavioral", "drivers", "trends",
}


def _topic_keyword(query: str) -> str:
    words = [w.strip(".,?!:;\"'") for w in (query or "").split()]
    words = [w for w in words if len(w) > 3 and w.lower() not in _STOPWORDS]
    return " ".join(words[:3]) if words else "this"

if TYPE_CHECKING:
    from agents.cognitive.types import AgentCognitiveOutput
    from nlp.topic_context import TopicContext


@dataclass
class CognitiveDraft:
    agent_id: str
    topic: str
    domain_label: str
    stimulus_snippet: str
    sentiment: float
    confidence: float
    behavioral_intent: str
    winning_voice: str
    key_concerns: list[str]
    emotions: dict[str, float]
    deliberation_summary: list[str]
    ocean_summary: str
    voice_register: str
    entropy: float
    memory_hooks: list[str] = field(default_factory=list)
    cluster_label: str = ""
    action_likelihood: float = 0.5

    def to_compact_json(self) -> str:
        payload = asdict(self)
        payload["key_concerns"] = payload["key_concerns"][:4]
        payload["deliberation_summary"] = payload["deliberation_summary"][:4]
        payload["memory_hooks"] = payload["memory_hooks"][:2]
        return json.dumps(payload, separators=(",", ":"))

    def to_prompt_brief(self) -> str:
        debate = compact_debate_lines(self.deliberation_summary, max_lines=4, max_line_chars=56)
        concerns = ", ".join(self.key_concerns[:4]) or "general uncertainty"
        mem = "; ".join(self.memory_hooks[:2])
        em = self.emotions
        lines = [
            f"Topic: {clamp_text(self.topic, 80)}",
            f"Domain: {self.domain_label}",
            f"Stimulus: {clamp_text(self.stimulus_snippet, 120)}",
            f"Sentiment: {self.sentiment:+.2f} | confidence: {self.confidence:.2f} | entropy: {self.entropy:.2f}",
            f"Intent: {clamp_text(self.behavioral_intent, 80)}",
            f"Dominant belief voice: {clamp_text(self.winning_voice, 40)}",
            f"Concerns: {concerns}",
            f"Emotions: trust={em.get('trust', 0):.2f} fear={em.get('fear', 0):.2f} "
            f"curiosity={em.get('curiosity', 0):.2f} excitement={em.get('excitement', 0):.2f}",
            f"Inner debate: {debate}",
            f"OCEAN: {self.ocean_summary} | voice: {self.voice_register}",
        ]
        if mem:
            lines.append(f"Memory hooks: {mem}")
        if self.cluster_label:
            lines.append(f"Cluster: {self.cluster_label}")
        return "\n".join(lines)

    def to_naturalize_brief(self) -> str:
        """Compact brief for fast Gemma naturalize (fewer input tokens, same semantics)."""
        debate = compact_debate_lines(self.deliberation_summary, max_lines=2, max_line_chars=44)
        concerns = ", ".join(self.key_concerns[:3]) or "general uncertainty"
        em = self.emotions
        dominant = max(
            ("fear", em.get("fear", 0)),
            ("curiosity", em.get("curiosity", 0)),
            ("trust", em.get("trust", 0)),
            key=lambda item: item[1],
        )[0]
        return "\n".join([
            f"Topic: {clamp_text(self.topic, 64)} | {self.domain_label}",
            f"Stimulus: {clamp_text(self.stimulus_snippet, 90)}",
            f"Sentiment {self.sentiment:+.2f} | intent: {clamp_text(self.behavioral_intent, 56)}",
            f"Voice: {clamp_text(self.winning_voice, 32)} | concerns: {concerns}",
            f"Emotion: {dominant} | debate: {debate}",
            f"{self.ocean_summary} | register: {self.voice_register}",
        ])


def _memory_hooks_from_state(memory: Any, limit: int = 2) -> list[str]:
    hooks: list[str] = []
    for pool in (memory.brand_experience, memory.past_campaigns, memory.social_exposure):
        for item in pool[:1]:
            if item and item not in hooks:
                hooks.append(str(item)[:80])
            if len(hooks) >= limit:
                return hooks
    return hooks


def _resolve_topic(topic_ctx: TopicContext | None, topic: str, stimulus: str) -> tuple[str, str]:
    domain = topic_ctx.domain_label if topic_ctx else "general"
    if topic_ctx and topic_ctx.keyphrases:
        t = topic_ctx.keyphrases[0]
    elif topic and topic != "this":
        t = topic
    else:
        t = _topic_keyword(stimulus)
    return t, domain


def build_cognitive_draft(
    output: AgentCognitiveOutput,
    topic_ctx: TopicContext | None,
    stimulus: str,
    *,
    topic: str = "",
    voice_register: str = "casual",
) -> CognitiveDraft:
    state = output.state
    ocean = state.ocean
    topic_str, domain = _resolve_topic(topic_ctx, topic, stimulus)
    ocean_summary = (
        f"O{round(ocean.O)} C{round(ocean.C)} E{round(ocean.E)} "
        f"A{round(ocean.A)} N{round(ocean.N)}"
    )
    return CognitiveDraft(
        agent_id=state.agent_id,
        topic=topic_str,
        domain_label=domain,
        stimulus_snippet=clamp_text(stimulus, 160),
        sentiment=round(output.sentiment, 3),
        confidence=round(state.confidence, 3),
        behavioral_intent=clamp_text(output.behavioral_intent, 120),
        winning_voice=state.deliberation.winning_voice or "Mixed",
        key_concerns=list(output.key_concerns[:5]),
        emotions=state.emotions.as_dict(),
        deliberation_summary=state.deliberation.summary_lines(),
        ocean_summary=ocean_summary,
        voice_register=voice_register,
        entropy=round(state.total_entropy, 3),
        memory_hooks=_memory_hooks_from_state(state.memory),
        cluster_label=state.cluster_label,
        action_likelihood=round(output.action_likelihood, 3),
    )


def build_draft_from_persona_fields(
    *,
    agent_id: str,
    ocean: OceanScores,
    sentiment: float,
    behavioral_intent: str,
    emotional_state: str,
    key_concerns: list[str],
    action_likelihood: float,
    cluster_label: str,
    top_facets: list[FacetScore],
    topic: str,
    stimulus: str = "",
    voice_register: str = "casual",
    entropy: float = 0.5,
    confidence: float = 0.5,
) -> CognitiveDraft:
    """Legacy psychometric path — draft from archetype fields without deliberation trace."""
    ocean_summary = (
        f"O{round(ocean.O)} C{round(ocean.C)} E{round(ocean.E)} "
        f"A{round(ocean.A)} N{round(ocean.N)}"
    )
    salient = [f.name for f in top_facets if f.band != "moderate"][:2]
    winning = salient[0] if salient else "General outlook"
    debate = [
        f"{f.name} ({f.band}): driver"
        for f in top_facets[:3]
    ]
    return CognitiveDraft(
        agent_id=agent_id,
        topic=topic or _topic_keyword(stimulus),
        domain_label="general",
        stimulus_snippet=clamp_text(stimulus or topic, 120),
        sentiment=round(sentiment, 3),
        confidence=round(confidence, 3),
        behavioral_intent=clamp_text(behavioral_intent, 120),
        winning_voice=winning,
        key_concerns=list(key_concerns[:5]) or ["general uncertainty"],
        emotions={
            "trust": 0.5,
            "fear": 0.4 if ocean.N > 55 else 0.25,
            "curiosity": 0.5 if ocean.O > 55 else 0.35,
            "excitement": 0.45 if ocean.E > 55 else 0.3,
        },
        deliberation_summary=debate,
        ocean_summary=ocean_summary,
        voice_register=voice_register,
        entropy=round(entropy, 3),
        memory_hooks=[f"emotional tone: {emotional_state}"] if emotional_state else [],
        cluster_label=cluster_label,
        action_likelihood=round(action_likelihood, 3),
    )


def draft_fallback_line(draft: CognitiveDraft) -> str:
    """Compositional one-liner when Gemma naturalize fails — no phrase banks."""
    lean = "positive" if draft.sentiment > 0.15 else "negative" if draft.sentiment < -0.15 else "mixed"
    concern = draft.key_concerns[0] if draft.key_concerns else "the overall fit"
    topic = draft.topic if draft.topic != "this" else "this"
    voice = draft.winning_voice or "my gut"
    return (
        f"When it comes to {topic}, {voice} is what stands out for me — "
        f"I'm leaning {lean} ({draft.sentiment:+.2f}), and {concern} is on my mind."
    )[:220]
