"""Internal belief-node deliberation (programmatic, no LLM)."""
from __future__ import annotations

import random
from dataclasses import dataclass

from agents.cognitive.bias_network import bias_boost_for_node
from agents.cognitive.emotion_engine import emotion_weight_for_node
from agents.cognitive.entropy import entropy_ambivalence_shift, jitter_weight
from agents.cognitive.types import BeliefUtterance, CognitiveStateVector, DeliberationTrace


@dataclass(frozen=True)
class BeliefNodeDef:
    node_id: str
    voice: str
    base_from_facets: str  # facet name driving base stance


_BELIEF_NODES: list[BeliefNodeDef] = [
    BeliefNodeDef("price_sensitivity", "Price Sensitivity", "Cautiousness"),
    BeliefNodeDef("trust_brand", "Trust", "Trust"),
    BeliefNodeDef("novelty_interest", "Novelty Interest", "Adventurousness"),
    BeliefNodeDef("social_proof", "Social Proof", "Gregariousness"),
    BeliefNodeDef("risk_aversion", "Risk Aversion", "Anxiety"),
    BeliefNodeDef("authority", "Authority", "Dutifulness"),
    BeliefNodeDef("status_quo", "Status Quo", "Orderliness"),
]


def _base_stance(state: CognitiveStateVector, facet_name: str) -> float:
    score = state.facets.get(facet_name, 50.0)
    return (score - 50.0) / 50.0


def _memory_stance_shift(state: CognitiveStateVector, node_id: str) -> float:
    mem = state.memory
    shift = mem.trust_modifier * 0.3
    if node_id == "trust_brand":
        return shift
    if node_id == "risk_aversion" and mem.has_negative_brand():
        return 0.25
    if node_id == "social_proof" and any("skeptical" in s.lower() for s in mem.social_exposure):
        return -0.15
    return 0.0


def deliberate(
    state: CognitiveStateVector,
    facet_weights: dict[str, float],
    evidence_polarity: float,
    rng: random.Random,
) -> tuple[float, float, float, list[str], str, DeliberationTrace]:
    """
    Run belief competition. Returns:
    sentiment, confidence, uncertainty, key_concerns, behavioral_intent, trace
    """
    utterances: list[BeliefUtterance] = []
    weighted_sum = 0.0
    weight_total = 0.0
    stances: list[float] = []

    for node in _BELIEF_NODES:
        base = _base_stance(state, node.base_from_facets)
        fw = facet_weights.get(node.base_from_facets, 1.0)
        em_w = emotion_weight_for_node(state, node.node_id)
        bias_w = bias_boost_for_node(state, node.node_id)
        mem_shift = _memory_stance_shift(state, node.node_id)

        raw_weight = (0.35 + em_w * 0.35 + bias_w + fw * 0.15) * (1.0 + state.total_entropy * 0.2)
        weight = jitter_weight(raw_weight, state.total_entropy, rng)
        stance = base + mem_shift + evidence_polarity * 0.15
        stance = max(-1.0, min(1.0, stance))

        utterances.append(
            BeliefUtterance(node_id=node.node_id, voice=node.voice, stance=round(stance, 3), weight=round(weight, 3))
        )
        weighted_sum += stance * weight
        weight_total += weight
        stances.append(stance)

    sentiment = weighted_sum / max(weight_total, 1e-6)
    conflict = float(max(stances) - min(stances)) if stances else 0.0
    confidence = max(0.1, min(0.95, 1.0 - conflict * 0.55 - state.total_entropy * 0.15))
    uncertainty = round(1.0 - confidence, 3)

    sentiment = entropy_ambivalence_shift(sentiment, confidence, state.total_entropy, rng)
    sentiment = round(max(-1.0, min(1.0, sentiment)), 3)

    winning = max(utterances, key=lambda u: u.weight * abs(u.stance))
    trace = DeliberationTrace(
        utterances=utterances,
        winning_voice=winning.voice,
        conflict_level=round(conflict, 3),
    )

    concerns = _derive_concerns(state, utterances, sentiment)
    intent = _derive_intent(sentiment, confidence, trace.winning_voice)
    return sentiment, confidence, uncertainty, concerns, intent, trace


def _derive_concerns(
    state: CognitiveStateVector,
    utterances: list[BeliefUtterance],
    sentiment: float,
) -> list[str]:
    concerns: list[str] = []
    by_id = {u.node_id: u for u in utterances}
    if by_id.get("price_sensitivity") and by_id["price_sensitivity"].stance < -0.1:
        concerns.append("cost and value")
    if by_id.get("risk_aversion") and by_id["risk_aversion"].stance > 0.1:
        concerns.append("downside risk")
    if by_id.get("trust_brand") and by_id["trust_brand"].stance < 0:
        concerns.append("trust and credibility")
    if by_id.get("social_proof") and by_id["social_proof"].stance < 0:
        concerns.append("lack of social validation")
    if state.emotions.fear > 0.6:
        concerns.append("uncertainty")
    if sentiment < 0 and "uncertainty" not in concerns:
        concerns.append("long-term fit")
    return concerns[:5] or ["general uncertainty"]


def _derive_intent(sentiment: float, confidence: float, winning_voice: str) -> str:
    if sentiment > 0.35 and confidence > 0.55:
        return "likely to engage or adopt"
    if sentiment > 0.1:
        return "interested but still evaluating"
    if sentiment < -0.25:
        return "unlikely to adopt without changes"
    if winning_voice == "Price Sensitivity":
        return "waiting for better value signals"
    if winning_voice == "Social Proof":
        return "would follow others' lead"
    return "undecided; needs more information"
