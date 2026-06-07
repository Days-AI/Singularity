"""Shared test fixtures for decision intelligence agents."""
from __future__ import annotations

from state import EvidenceItem, OceanScores, PersonaOpinion, PersonaResponse, SingularityState


def make_state(**overrides) -> SingularityState:
    base = SingularityState(query="How will audiences react to our new product launch?", flow_uuid="test-uuid")
    opinions = []
    for i in range(30):
        cluster = i % 3
        sent = -0.4 if cluster == 0 else 0.1 if cluster == 1 else 0.5
        opinions.append(PersonaOpinion(
            id=f"p_{i}",
            archetype_id=f"a_{cluster}",
            cluster=cluster,
            cluster_label=["Skeptics", "Pragmatists", "Enthusiasts"][cluster],
            ocean=OceanScores(O=50 + cluster * 10, C=55, E=45, A=50, N=40 - cluster * 5),
            sentiment=sent,
            behavioral_intent="consider" if sent > 0 else "wait",
            emotional_state="cautious" if sent < 0 else "interested",
            key_concerns=["price", "trust"] if cluster == 0 else ["value"],
            action_likelihood=0.3 + cluster * 0.2,
            comment="test comment",
        ))
    responses = [
        PersonaResponse(
            archetype_id=f"a_{i % 3}",
            ocean=OceanScores(O=50, C=55, E=45, A=50, N=40),
            sentiment_score=-0.2 + (i % 3) * 0.35,
            action_likelihood=0.3 + (i % 3) * 0.2,
        )
        for i in range(6)
    ]
    evidence = [
        EvidenceItem(source="test", title="Market buzz rising", detail="Positive trend", sentiment=0.3),
        EvidenceItem(source="test", title="Competitor launch", detail="Crowded space", sentiment=-0.1),
    ]
    state = base.model_copy(update={
        "persona_opinions": opinions,
        "persona_responses": responses,
        "evidence": evidence,
        "ocean_mean": OceanScores(O=55, C=55, E=48, A=52, N=38),
    })
    if overrides:
        state = state.model_copy(update=overrides)
    return state


def run_pipeline_metrics(state: SingularityState) -> SingularityState:
    """Run decision-intelligence stages sequentially to populate metrics."""
    from agents import prediction_market, monte_carlo, swarm_optimization

    pm = prediction_market.run(state)
    state.metrics["prediction_market"] = prediction_market.to_metrics(pm)
    mc = monte_carlo.run(state)
    state.metrics["monte_carlo"] = monte_carlo.to_metrics(mc)
    sw = swarm_optimization.run(state)
    state.metrics["swarm_optimization"] = swarm_optimization.to_metrics(sw)
    return state
