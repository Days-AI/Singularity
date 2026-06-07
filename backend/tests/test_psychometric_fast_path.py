from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from agents import ipip
from agents.psychometric import _programmatic_archetype_responses, run
from state import OceanScores, SingularityState


def test_programmatic_archetype_responses_count():
    ocean = np.array([[50, 55, 45, 52, 38], [60, 50, 70, 48, 30]], dtype=float)
    facets = np.full((2, 30), 50.0)
    pop = ipip.IpipPopulation(ocean=ocean, facets=facets)
    labels = np.array([0, 1])
    archetypes = [
        OceanScores(O=50, C=55, E=45, A=52, N=38),
        OceanScores(O=60, C=50, E=70, A=48, N=30),
    ]
    responses = _programmatic_archetype_responses(pop, labels, archetypes)
    assert len(responses) == 2
    assert all(r.facets for r in responses)
    assert responses[0].validated_ocean is not None


@pytest.mark.parametrize("n_arch", [36])
def test_skip_archetype_llm_when_cognitive_ipip(monkeypatch, n_arch):
    monkeypatch.setenv("COGNITIVE_AGENTS_ENABLED", "true")
    monkeypatch.setenv("PERSONA_ARCHETYPES", str(n_arch))
    from config import get_settings

    get_settings.cache_clear()

    ocean = np.random.default_rng(0).uniform(20, 80, (150, 5))
    facets = np.random.default_rng(1).uniform(20, 80, (150, 30))
    fake_pop = ipip.IpipPopulation(ocean=ocean, facets=facets)

    simulate_mock = AsyncMock()

    async def _fake_cognitive(**kwargs):
        from agents.cognitive.state_engine import PopulationCognitiveResult
        from agents.cognitive.types import AgentCognitiveOutput, CognitiveStateVector

        outputs = []
        for i in range(kwargs["population"]):
            oc = OceanScores(
                O=float(kwargs["pop_ocean"][i, 0]),
                C=float(kwargs["pop_ocean"][i, 1]),
                E=float(kwargs["pop_ocean"][i, 2]),
                A=float(kwargs["pop_ocean"][i, 3]),
                N=float(kwargs["pop_ocean"][i, 4]),
            )
            state = CognitiveStateVector(
                agent_id=f"p_{i}",
                ocean=oc,
                facets={},
                entropy_seed=i,
            )
            outputs.append(
                AgentCognitiveOutput(
                    state=state,
                    sentiment=float(kwargs["pop_sent"][i]),
                    behavioral_intent="wait",
                    key_concerns=["test"],
                    action_likelihood=0.5,
                    comment="programmatic",
                )
            )
        return PopulationCognitiveResult(outputs=outputs)

    batches: list = []

    async def emit_batch(payload):
        batches.append(payload)

    with patch("agents.psychometric.ipip.load", return_value=fake_pop):
        with patch("agents.psychometric._simulate_archetype", simulate_mock):
            with patch(
                "agents.psychometric.run_population_cognitive",
                side_effect=_fake_cognitive,
            ):
                state = SingularityState(query="test launch", flow_uuid="fast-path")
                result = asyncio.run(run(state, emit_batch))

    simulate_mock.assert_not_called()
    assert result.population == 150
    assert len(result.opinions) == 150
    assert len(batches) >= 1


def test_archetype_llm_runs_when_cognitive_disabled(monkeypatch):
    monkeypatch.setenv("COGNITIVE_AGENTS_ENABLED", "false")
    monkeypatch.setenv("PERSONA_ARCHETYPES", "2")
    from config import get_settings

    get_settings.cache_clear()

    simulate_mock = AsyncMock(
        side_effect=lambda arch_id, ocean, context, stimulus: __import__(
            "agents.psychometric", fromlist=["PersonaResponse"]
        ).PersonaResponse(
            archetype_id=arch_id,
            ocean=ocean,
            sentiment_score=0.1,
        )
    )

    async def emit_batch(_payload):
        pass

    with patch("agents.psychometric.ipip.load", return_value=None):
        with patch("agents.psychometric._simulate_archetype", simulate_mock):
            state = SingularityState(query="test", flow_uuid="legacy")
            asyncio.run(run(state, emit_batch))

    assert simulate_mock.await_count == 2
