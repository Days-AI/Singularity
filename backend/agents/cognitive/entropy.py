"""Entropy injection for non-deterministic persona responses."""
from __future__ import annotations

import random

import numpy as np

from agents.cognitive.types import CognitiveStateVector, EmotionalState
from state import OceanScores


def compute_entropy_seed(agent_index: int, run_seed: int = 0) -> int:
    return (agent_index * 7919 + run_seed * 104729) & 0x7FFFFFFF


def personality_entropy(ocean: OceanScores, facets: dict[str, float]) -> float:
    dims = [ocean.O, ocean.C, ocean.E, ocean.A, ocean.N]
    variance = float(np.var(dims)) / 2500.0
    extremity = sum(abs(v - 50.0) for v in facets.values()) / max(1, len(facets) * 50.0)
    return min(1.0, 0.3 * variance + 0.7 * extremity)


def emotional_entropy(emotions: EmotionalState) -> float:
    vals = [emotions.trust, emotions.fear, emotions.curiosity, emotions.excitement]
    spread = float(np.std(vals))
    return min(1.0, spread * 2.5 + emotions.fear * 0.2)


def contextual_entropy(evidence_polarity: float, social_divergence: float = 0.0) -> float:
    return min(1.0, abs(evidence_polarity) * 0.4 + social_divergence * 0.6)


def total_entropy(
    state: CognitiveStateVector,
    evidence_polarity: float = 0.0,
    social_divergence: float = 0.0,
) -> float:
    p = personality_entropy(state.ocean, state.facets)
    e = emotional_entropy(state.emotions)
    c = contextual_entropy(evidence_polarity, social_divergence)
    return round(min(1.0, 0.4 * p + 0.35 * e + 0.25 * c), 4)


def jitter_weight(base: float, entropy: float, rng: random.Random) -> float:
    noise = rng.uniform(-entropy * 0.35, entropy * 0.35)
    return max(0.05, base + noise)


def entropy_ambivalence_shift(sentiment: float, confidence: float, entropy: float, rng: random.Random) -> float:
    """When confidence is near 0.5, entropy can flip stance slightly."""
    if confidence > 0.65 or confidence < 0.35:
        return sentiment
    if rng.random() < entropy * 0.4:
        return sentiment + rng.uniform(-0.25, 0.25)
    return sentiment
