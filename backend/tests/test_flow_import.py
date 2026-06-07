from __future__ import annotations


def test_flow_imports():
    import flow  # noqa: F401

    from agents import prediction_market, monte_carlo, swarm_optimization, decision_engine

    assert callable(prediction_market.run)
    assert callable(monte_carlo.run)
    assert callable(swarm_optimization.run)
    assert callable(decision_engine.run)


def test_singularity_engine_prompt_loaded():
    from prompts import SINGULARITY_ENGINE_SYSTEM

    assert "Synthetic Society Simulation Engine" in SINGULARITY_ENGINE_SYSTEM
    assert "Never provide a single answer" in SINGULARITY_ENGINE_SYSTEM
