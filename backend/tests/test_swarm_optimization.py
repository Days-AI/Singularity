from __future__ import annotations

from agents import swarm_optimization
from fixtures import make_state, run_pipeline_metrics


def test_swarm_communications_domain():
    state = make_state(query="How should our PR communications strategy handle this crisis?")
    state = run_pipeline_metrics(state)
    metrics = swarm_optimization.to_metrics(swarm_optimization.run(state))

    assert metrics["domain"] == "communications"
    assert metrics["algorithm"] == "aco"
    assert "steps" in metrics["optimal_path"]
    assert metrics["convergence_iterations"] == 50


def test_swarm_marketing_domain():
    state = make_state(query="Optimize our digital marketing campaign headlines")
    metrics = swarm_optimization.to_metrics(swarm_optimization.run(state))

    assert metrics["domain"] == "marketing"
    assert metrics["algorithm"] == "pso"
    assert metrics["optimal_path"]["score"] >= 0.0


def test_swarm_alternatives_sorted():
    state = run_pipeline_metrics(make_state())
    metrics = swarm_optimization.to_metrics(swarm_optimization.run(state))
    scores = [a["score"] for a in metrics["alternatives"]]
    assert scores == sorted(scores, reverse=True)
