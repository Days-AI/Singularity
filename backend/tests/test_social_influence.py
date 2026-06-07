"""Tests for social influence propagation."""
from __future__ import annotations

from agents.cognitive.social_influence import (
    InfluencerRecord,
    apply_social_influence,
    compute_social_position,
)
from agents.cognitive.types import AgentCognitiveOutput, CognitiveStateVector
from state import OceanScores
import numpy as np


def _output(idx: int, sentiment: float, cluster: int, social: float) -> AgentCognitiveOutput:
    state = CognitiveStateVector(
        agent_id=f"p_{idx:04d}",
        ocean=OceanScores(O=50, C=50, E=70 if social > 0.5 else 40, A=50, N=50),
        facets={"Assertiveness": 80.0 if social > 0.5 else 40.0},
        social_position=social,
        cluster=cluster,
        cluster_label="Enthusiasts" if cluster == 2 else "Pragmatists",
    )
    return AgentCognitiveOutput(
        state=state,
        sentiment=sentiment,
        behavioral_intent="evaluating",
        key_concerns=[],
        action_likelihood=0.5,
    )


def test_influencer_shifts_followers_in_cluster():
    outputs = [
        _output(0, 0.6, 1, 0.8),
        _output(1, 0.0, 1, 0.2),
        _output(2, 0.0, 1, 0.2),
        _output(3, 0.1, 2, 0.3),
    ]
    clusters = np.array([1, 1, 1, 2])
    pop_ocean = np.array([[50, 50, 70, 50, 50], [50, 50, 40, 50, 50], [50, 50, 40, 50, 50], [50, 50, 40, 50, 50]])
    pop_facets = np.array([[50] * 30, [50] * 30, [50] * 30, [50] * 30])
    influencers = [InfluencerRecord(agent_index=0, support=0.6, reach=0.7)]
    idx = apply_social_influence(outputs, clusters, influencers, contagion_strength=0.22)
    assert outputs[1].sentiment > 0.0
    assert idx > 0


def test_social_position_scales_with_extraversion():
    ocean = np.array([50, 50, 80, 50, 50])
    facets = np.zeros(30)
    names = ["Assertiveness", "Gregariousness"]
    pos = compute_social_position(ocean, facets, names)
    assert 0.0 <= pos <= 1.0
