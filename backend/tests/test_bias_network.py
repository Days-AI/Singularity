"""Tests for facet-conditioned bias activation."""
from __future__ import annotations

from agents.cognitive.bias_network import activate_biases
from agents.cognitive.memory_store import generate_memory
from agents.cognitive.types import CognitiveStateVector
from state import OceanScores


def _state(intellect: float = 50.0, anxiety: float = 50.0, negative_brand: bool = False) -> CognitiveStateVector:
    facets = {"Intellect": intellect, "Anxiety": anxiety, "Trust": 50.0}
    memory = generate_memory(42, "arch_01")
    if negative_brand:
        memory.brand_experience = ["negative customer service experience"]
    return CognitiveStateVector(
        agent_id="p_0042",
        ocean=OceanScores(O=50, C=50, E=50, A=50, N=anxiety),
        facets=facets,
        memory=memory,
    )


def test_high_intellect_activates_complexity_biases():
    state = _state(intellect=92.0)
    active = activate_biases(state)
    ids = {b.bias_id for b in active}
    assert "complexity_preference" in ids or "availability_bias" in ids


def test_high_anxiety_activates_loss_aversion():
    state = _state(anxiety=88.0)
    active = activate_biases(state)
    ids = {b.bias_id for b in active}
    assert "loss_aversion" in ids or "negativity_bias" in ids


def test_negative_brand_boosts_confirmation():
    state = _state(intellect=40.0, negative_brand=True)
    active = activate_biases(state)
    ids = {b.bias_id for b in active}
    assert "confirmation_bias" in ids or "loss_aversion" in ids
