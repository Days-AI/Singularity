"""Facet- and memory-conditioned cognitive bias activation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from agents.cognitive.types import ActiveBias, AgentMemory, CognitiveStateVector

BiasRule = Callable[[CognitiveStateVector], float]


@dataclass(frozen=True)
class BiasDefinition:
    bias_id: str
    label: str
    rule: BiasRule
    deliberation_targets: tuple[str, ...]  # belief node ids boosted


def _facet_high(state: CognitiveStateVector, name: str, thresh: float = 0.66) -> float:
    return max(0.0, (state.facet_norm(name) - thresh) / max(1e-6, 1.0 - thresh))


def _facet_low(state: CognitiveStateVector, name: str, thresh: float = 0.34) -> float:
    return max(0.0, (thresh - state.facet_norm(name)) / max(1e-6, thresh))


BIAS_DEFINITIONS: list[BiasDefinition] = [
    BiasDefinition(
        "confirmation_bias",
        "Confirmation Bias",
        lambda s: 0.4 * _facet_low(s, "Intellect") + (0.3 if s.memory.has_negative_brand() else 0.0),
        ("trust_brand", "status_quo"),
    ),
    BiasDefinition(
        "loss_aversion",
        "Loss Aversion",
        lambda s: 0.5 * _facet_high(s, "Anxiety") + 0.3 * _facet_high(s, "Cautiousness"),
        ("risk_aversion", "price_sensitivity"),
    ),
    BiasDefinition(
        "authority_bias",
        "Authority Bias",
        lambda s: 0.4 * _facet_high(s, "Dutifulness") + 0.2 * _facet_low(s, "Liberalism"),
        ("authority", "trust_brand"),
    ),
    BiasDefinition(
        "bandwagon",
        "Bandwagon Effect",
        lambda s: 0.5 * _facet_high(s, "Gregariousness") + 0.2 * _facet_high(s, "Excitement"),
        ("social_proof", "novelty_interest"),
    ),
    BiasDefinition(
        "availability_bias",
        "Availability Bias",
        lambda s: 0.35 * _facet_high(s, "Imagination") + 0.25 * _facet_high(s, "Activity"),
        ("novelty_interest", "social_proof"),
    ),
    BiasDefinition(
        "anchoring",
        "Anchoring Bias",
        lambda s: 0.4 * _facet_high(s, "Orderliness") + 0.2 * _facet_high(s, "Self-Efficacy"),
        ("price_sensitivity", "status_quo"),
    ),
    BiasDefinition(
        "status_quo",
        "Status Quo Bias",
        lambda s: 0.45 * _facet_low(s, "Adventurousness") + 0.25 * _facet_high(s, "Cautiousness"),
        ("status_quo", "risk_aversion"),
    ),
    BiasDefinition(
        "halo_effect",
        "Halo Effect",
        lambda s: 0.5 * _facet_high(s, "Trust") + 0.2 * _facet_high(s, "Cheerfulness"),
        ("trust_brand", "authority"),
    ),
    BiasDefinition(
        "recency_bias",
        "Recency Bias",
        lambda s: 0.35 * _facet_high(s, "Activity") + 0.25 * _facet_high(s, "Excitement"),
        ("novelty_interest", "social_proof"),
    ),
    BiasDefinition(
        "social_proof",
        "Social Proof",
        lambda s: 0.5 * _facet_high(s, "Friendliness") + 0.3 * _facet_high(s, "Self-Consciousness"),
        ("social_proof",),
    ),
    BiasDefinition(
        "negativity_bias",
        "Negativity Bias",
        lambda s: 0.45 * _facet_high(s, "Anxiety") + 0.25 * _facet_high(s, "Anger"),
        ("risk_aversion", "price_sensitivity"),
    ),
    BiasDefinition(
        "complexity_preference",
        "Complexity Preference",
        lambda s: 0.55 * _facet_high(s, "Intellect") + 0.25 * _facet_high(s, "Imagination"),
        ("novelty_interest", "trust_brand"),
    ),
]


def activate_biases(state: CognitiveStateVector) -> list[ActiveBias]:
    """Compute active biases from facets + memory."""
    active: list[ActiveBias] = []
    for defn in BIAS_DEFINITIONS:
        raw = defn.rule(state)
        if state.memory.has_negative_brand() and defn.bias_id in (
            "confirmation_bias",
            "loss_aversion",
            "negativity_bias",
        ):
            raw = min(1.0, raw + 0.15)
        strength = max(0.0, min(1.0, raw))
        if strength >= 0.12:
            active.append(ActiveBias(bias_id=defn.bias_id, strength=round(strength, 3)))
    active.sort(key=lambda b: b.strength, reverse=True)
    return active[:8]


def bias_boost_for_node(state: CognitiveStateVector, node_id: str) -> float:
    """Aggregate deliberation weight boost from active biases for a belief node."""
    boost = 0.0
    for bias in state.active_biases:
        for defn in BIAS_DEFINITIONS:
            if defn.bias_id == bias.bias_id and node_id in defn.deliberation_targets:
                boost += bias.strength * 0.35
    return boost
