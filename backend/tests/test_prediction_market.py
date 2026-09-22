from __future__ import annotations

from agents import causal, prediction_market
from fixtures import make_state, run_pipeline_metrics


def test_prediction_market_outcome_bounds():
    state = run_pipeline_metrics(make_state())
    metrics = prediction_market.to_metrics(prediction_market.run(state))

    assert 0.0 <= metrics["overall_outcome"] <= 100.0
    assert metrics["confidence_interval"][0] <= metrics["confidence_interval"][1]
    assert len(metrics["forecasts"]) == 8
    for f in metrics["forecasts"]:
        assert 0.0 <= f["expected"] <= 100.0
        assert f["ci_low"] <= f["ci_high"]


def test_prediction_market_feeds_causal():
    state = run_pipeline_metrics(make_state())
    graph = causal.build(state)
    assert 0.0 <= graph.overall_prediction <= 100.0


def test_prediction_market_string_cluster_keys_from_deliberation():
    """Production cognitive path stores string cluster keys in deliberation."""
    state = make_state()
    state.metrics["deliberation"] = {
        "cluster_sentiments": {"Skeptics": -0.4, "Pragmatists": 0.1, "Enthusiasts": 0.5},
        "cluster_actions": {"Skeptics": 0.25, "Pragmatists": 0.5, "Enthusiasts": 0.75},
        "confidence_score": 0.65,
        "mean_sentiment": 0.05,
        "mean_action_likelihood": 0.5,
    }
    metrics = prediction_market.to_metrics(prediction_market.run(state))
    assert 0.0 <= metrics["overall_outcome"] <= 100.0
    assert len(metrics["forecasts"]) == 8
