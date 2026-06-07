"""Tests for entropy engine variance and reproducibility."""
from __future__ import annotations

import random

from agents.cognitive.deliberation import deliberate
from agents.cognitive.entropy import (
    compute_entropy_seed,
    jitter_weight,
    total_entropy,
)
from agents.cognitive.types import CognitiveStateVector, EmotionalState
from state import OceanScores


def _minimal_state(seed: int) -> CognitiveStateVector:
    facets = {f"F{i}": 50.0 + (i % 5) * 8 for i in range(30)}
    return CognitiveStateVector(
        agent_id="p_0001",
        ocean=OceanScores(O=70, C=45, E=55, A=60, N=40),
        facets=facets,
        emotions=EmotionalState(trust=0.6, fear=0.3, curiosity=0.7, excitement=0.5),
        entropy_seed=seed,
        total_entropy=0.45,
    )


def test_entropy_seed_reproducible():
    assert compute_entropy_seed(10, 42) == compute_entropy_seed(10, 42)
    assert compute_entropy_seed(10, 42) != compute_entropy_seed(11, 42)


def test_total_entropy_in_range():
    state = _minimal_state(1)
    e = total_entropy(state, evidence_polarity=0.2)
    assert 0.0 <= e <= 1.0


def test_same_seed_same_deliberation_sentiment():
    state = _minimal_state(100)
    weights = {"Intellect": 1.2, "Trust": 1.0}
    rng1 = random.Random(100)
    s1, _, _, _, _, _ = deliberate(state, weights, 0.0, rng1)
    rng2 = random.Random(100)
    s2, _, _, _, _, _ = deliberate(state, weights, 0.0, rng2)
    assert s1 == s2


def test_entropy_jitter_varies_by_seed():
    rng1 = random.Random(1)
    rng2 = random.Random(2)
    w1 = jitter_weight(0.5, 0.8, rng1)
    w2 = jitter_weight(0.5, 0.8, rng2)
    assert w1 != w2
