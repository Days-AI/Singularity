"""Continuous affect model: trust, fear, curiosity, excitement."""
from __future__ import annotations

from agents.cognitive.types import ActiveBias, AgentMemory, CognitiveStateVector, EmotionalState
from state import OceanScores


def init_emotions(ocean: OceanScores, facets: dict[str, float], memory: AgentMemory) -> EmotionalState:
    trust = 0.35 + ocean.A / 200.0 + facets.get("Trust", 50.0) / 200.0
    fear = 0.15 + ocean.N / 150.0 + facets.get("Anxiety", 50.0) / 200.0
    curiosity = 0.25 + ocean.O / 150.0 + facets.get("Intellect", 50.0) / 200.0
    excitement = 0.2 + ocean.E / 150.0 + facets.get("Excitement", 50.0) / 200.0
    trust += memory.trust_modifier * 0.2
    trust = _clamp(trust)
    fear = _clamp(fear)
    curiosity = _clamp(curiosity)
    excitement = _clamp(excitement)
    return EmotionalState(trust=trust, fear=fear, curiosity=curiosity, excitement=excitement)


def update_emotions_after_response(
    emotions: EmotionalState,
    sentiment: float,
    active_biases: list[ActiveBias],
    memory: AgentMemory,
) -> EmotionalState:
    """Shift affect after deliberation + response."""
    e = EmotionalState(
        trust=emotions.trust,
        fear=emotions.fear,
        curiosity=emotions.curiosity,
        excitement=emotions.excitement,
    )
    e.trust += sentiment * 0.08
    e.fear += max(0.0, -sentiment) * 0.1
    e.curiosity += max(0.0, sentiment) * 0.06
    e.excitement += sentiment * 0.07
    if any(b.bias_id == "negativity_bias" for b in active_biases):
        e.fear = min(1.0, e.fear + 0.05)
    if memory.has_negative_brand():
        e.trust = max(0.0, e.trust - 0.06)
    return EmotionalState(
        trust=_clamp(e.trust),
        fear=_clamp(e.fear),
        curiosity=_clamp(e.curiosity),
        excitement=_clamp(e.excitement),
    )


def emotion_weight_for_node(state: CognitiveStateVector, node_id: str) -> float:
    """Map emotions to belief-node influence."""
    em = state.emotions
    mapping = {
        "trust_brand": em.trust,
        "price_sensitivity": em.fear * 0.7 + (1 - em.trust) * 0.3,
        "novelty_interest": em.curiosity + em.excitement * 0.5,
        "social_proof": em.trust * 0.4 + em.excitement * 0.3,
        "risk_aversion": em.fear,
        "authority": em.trust * 0.6,
        "status_quo": (1 - em.curiosity) * 0.5 + em.fear * 0.3,
    }
    return mapping.get(node_id, 0.4)


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))
