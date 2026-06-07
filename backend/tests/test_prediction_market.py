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
