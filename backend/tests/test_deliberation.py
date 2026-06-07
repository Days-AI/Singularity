"""Tests for internal deliberation outcomes."""
from __future__ import annotations

import random

from agents.cognitive.bias_network import activate_biases
from agents.cognitive.deliberation import deliberate
from agents.cognitive.memory_store import generate_memory
from agents.cognitive.types import CognitiveStateVector
from state import OceanScores


def test_high_anxiety_yields_cautious_sentiment():
    facets = {
        "Anxiety": 90.0,
        "Cautiousness": 85.0,
        "Trust": 30.0,
        "Intellect": 55.0,
        "Gregariousness": 40.0,
        "Adventurousness": 35.0,
        "Dutifulness": 50.0,
        "Orderliness": 60.0,
    }
    state = CognitiveStateVector(
        agent_id="p_0002",
        ocean=OceanScores(O=40, C=70, E=40, A=45, N=85),
        facets=facets,
        memory=generate_memory(2, "arch_00"),
        total_entropy=0.3,
    )
    state.active_biases = activate_biases(state)
    rng = random.Random(7)
    sentiment, confidence, uncertainty, concerns, intent, trace = deliberate(
        state, {k: 1.0 for k in facets}, -0.1, rng
    )
    assert sentiment < 0.35
    assert any("risk" in c.lower() or "trust" in c.lower() or "uncertainty" in c.lower() for c in concerns)
    assert len(trace.utterances) >= 4


def test_high_openness_can_lean_positive_on_novelty():
    facets = {
        "Intellect": 88.0,
        "Adventurousness": 90.0,
        "Imagination": 85.0,
        "Anxiety": 25.0,
        "Trust": 65.0,
        "Gregariousness": 60.0,
        "Cautiousness": 30.0,
        "Dutifulness": 45.0,
        "Orderliness": 40.0,
    }
    state = CognitiveStateVector(
        agent_id="p_0003",
        ocean=OceanScores(O=85, C=45, E=60, A=55, N=25),
        facets=facets,
        memory=generate_memory(3, "arch_01"),
        total_entropy=0.35,
    )
    state.active_biases = activate_biases(state)
    rng = random.Random(3)
    sentiment, _, _, _, _, _ = deliberate(state, {k: 1.2 for k in facets}, 0.15, rng)
    assert sentiment > -0.1
