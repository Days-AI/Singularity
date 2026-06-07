"""Integration tests for cognitive population pipeline."""
from __future__ import annotations

import asyncio
from unittest.mock import patch

import numpy as np

from agents.cognitive.aggregate import aggregate_deliberation_metrics
from agents.cognitive.response_renderer import render_population_comments
from agents.cognitive.state_engine import run_population_cognitive
from agents.cognitive.types import AgentCognitiveOutput, CognitiveStateVector
from state import OceanScores, PersonaResponse


def test_population_cognitive_produces_deliberation_metrics():
    n = 24
    pop_ocean = np.random.default_rng(1).uniform(20, 80, (n, 5))
    pop_facets = np.random.default_rng(2).uniform(20, 80, (n, 30))
    pop_sent = np.zeros(n)
    assign = np.arange(n) % 3
    visual = np.array([i % 3 for i in range(n)])
    cluster_labels = {0: "Skeptics", 1: "Pragmatists", 2: "Enthusiasts"}
    responses = [
        PersonaResponse(
            archetype_id=f"arch_{i:02d}",
            ocean=OceanScores(O=50, C=50, E=50, A=50, N=50),
            facets={},
            sentiment_score=0.1,
        )
        for i in range(3)
    ]

    async def _fill_comments(outputs, llm_indices, *args, **kwargs):
        for i, out in enumerate(outputs):
            out.comment = f"comment {i}"
            out.response_source = "llm" if i in llm_indices else "programmatic"

    async def _run():
        with patch(
            "agents.cognitive.state_engine.render_population_comments",
            side_effect=_fill_comments,
        ):
            return await run_population_cognitive(
                population=n,
                pop_ocean=pop_ocean,
                pop_facets=pop_facets,
                pop_sent=pop_sent,
                archetype_assign=assign,
                responses=responses,
                visual_clusters=visual,
                cluster_labels=cluster_labels,
                query="Would you buy this product?",
                evidence=[],
                context="test context",
                stimulus="Would you buy this product?",
                topic="product",
                run_seed=42,
            )

    result = asyncio.run(_run())
    metrics = aggregate_deliberation_metrics(result.outputs, responses)
    assert metrics["agreement_rate"] >= 0.0
    assert "polarization_index" in metrics
    assert metrics["narrative_clusters"]
    assert len(result.outputs) == n
    assert all(o.comment for o in result.outputs)


def test_render_population_skips_llm_when_empty_sample():
    state = CognitiveStateVector(
        agent_id="p_0000",
        ocean=OceanScores(O=50, C=50, E=50, A=50, N=50),
        facets={"Trust": 50.0},
        entropy_seed=1,
        total_entropy=0.4,
    )
    out = AgentCognitiveOutput(
        state=state,
        sentiment=0.1,
        behavioral_intent="wait",
        key_concerns=["cost"],
        action_likelihood=0.4,
    )
    asyncio.run(render_population_comments([out], set(), "stimulus", "ctx", "topic", 0))
    assert out.comment
    assert out.response_source == "programmatic"
