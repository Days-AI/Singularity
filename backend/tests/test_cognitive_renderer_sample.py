"""Stratified LLM sample selection tests."""
from __future__ import annotations

import numpy as np

from agents.cognitive.response_renderer import select_llm_sample_indices
from agents.cognitive.types import AgentCognitiveOutput, CognitiveStateVector
from state import OceanScores


def _make_output(i: int, sent: float, entropy: float, cluster: int) -> AgentCognitiveOutput:
    state = CognitiveStateVector(
        agent_id=f"p_{i:04d}",
        ocean=OceanScores(O=50, C=50, E=50, A=50, N=50),
        facets={},
        total_entropy=entropy,
        cluster=cluster,
        cluster_label="Pragmatists",
    )
    return AgentCognitiveOutput(
        state=state,
        sentiment=sent,
        behavioral_intent="evaluate",
        key_concerns=[],
        action_likelihood=0.5,
    )


def test_stratified_sample_size():
    outputs = [_make_output(i, (i % 10) / 10 - 0.5, i / 100, i % 3) for i in range(100)]
    clusters = np.array([i % 3 for i in range(100)])
    chosen = select_llm_sample_indices(outputs, clusters, 50)
    assert len(chosen) == 50
