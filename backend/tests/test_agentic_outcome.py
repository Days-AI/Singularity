"""Agentic population signals must drive headline outcome probability."""
from __future__ import annotations

from agents import causal, prediction_market
from agents.cognitive.aggregate import aggregate_deliberation_metrics
from agents.cognitive.types import AgentCognitiveOutput, CognitiveStateVector
from fixtures import make_state, run_pipeline_metrics
from report.generate import _structured_data
from state import OceanScores, PersonaOpinion, PersonaResponse


def _make_cognitive_output(sentiment: float, action: float, label: str) -> AgentCognitiveOutput:
    ocean = OceanScores(O=50, C=50, E=50, A=50, N=50)
    state = CognitiveStateVector(
        agent_id="p_0001",
        ocean=ocean,
        facets={},
        cluster=0,
        cluster_label=label,
        archetype_id="arch_00",
        entropy_seed=1,
    )
    return AgentCognitiveOutput(
        state=state,
        sentiment=sentiment,
        behavioral_intent="consider",
        key_concerns=["price"],
        action_likelihood=action,
    )


def test_prediction_market_string_cluster_keys():
    """Cognitive deliberation uses string cluster labels — market must not crash."""
    outputs = [
        _make_cognitive_output(-0.5, 0.2, "Skeptics"),
        _make_cognitive_output(0.1, 0.5, "Pragmatists"),
        _make_cognitive_output(0.6, 0.8, "Enthusiasts"),
    ]
    delib = aggregate_deliberation_metrics(outputs, [])
    state = make_state()
    state.metrics["deliberation"] = delib

    metrics = prediction_market.to_metrics(prediction_market.run(state))
    assert 0.0 <= metrics["overall_outcome"] <= 100.0
    assert metrics["overall_outcome"] < 55.0  # skeptics pull down vs neutral 50


def test_outcome_uses_persona_opinions_over_archetypes():
    """Headline sentiment should follow 1500-agent opinions, not 36 archetype seeds."""
    state = make_state()
    # Archetypes mildly positive
    state.persona_responses = [
        PersonaResponse(
            archetype_id="a_0",
            ocean=OceanScores(O=60, C=55, E=50, A=52, N=40),
            sentiment_score=0.3,
            action_likelihood=0.6,
        )
    ]
    # Population strongly negative
    state.persona_opinions = [
        PersonaOpinion(
            id=f"p_{i}",
            archetype_id="a_0",
            cluster=0,
            cluster_label="Skeptics",
            ocean=OceanScores(O=50, C=50, E=50, A=50, N=50),
            sentiment=-0.7,
            behavioral_intent="reject",
            emotional_state="cautious",
            key_concerns=["trust"],
            action_likelihood=0.15,
            comment="skeptical",
        )
        for i in range(20)
    ]
    state.metrics["deliberation"] = {
        "mean_sentiment": -0.7,
        "mean_action_likelihood": 0.15,
        "cluster_sentiments": {"Skeptics": -0.7},
        "cluster_actions": {"Skeptics": 0.15},
        "confidence_score": 0.6,
        "agreement_rate": 0.8,
        "polarization_index": 0.1,
    }
    state = run_pipeline_metrics(state)

    archetype_only = causal.compute_outcome_probability(
        state.model_copy(update={"persona_opinions": [], "metrics": {}})
    )
    agentic = causal.compute_outcome_probability(state)
    assert agentic < archetype_only - 5.0


def test_outcome_breakdown_populated():
    state = run_pipeline_metrics(make_state())
    graph = causal.build(state)
    assert graph.outcome_breakdown
    assert "agentic_sentiment" in graph.outcome_breakdown or "market" in graph.outcome_breakdown
    assert graph.outcome_breakdown.get("headline") == graph.overall_prediction


def test_report_matches_causal_with_agentic_deliberation():
    state = run_pipeline_metrics(make_state())
    outputs = [_make_cognitive_output(0.4, 0.7, "Enthusiasts") for _ in range(5)]
    state.metrics["deliberation"] = aggregate_deliberation_metrics(outputs, [])
    state.metrics["prediction_market"] = prediction_market.to_metrics(
        prediction_market.run(state)
    )
    graph = causal.build(state)
    state.causal = graph
    data = _structured_data(state)
    assert data["outcome_probability"] == graph.overall_prediction
