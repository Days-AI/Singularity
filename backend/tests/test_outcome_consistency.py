"""Outcome probability must stay consistent across causal, report data, and refresh."""
from __future__ import annotations

from agents import causal, forecast, prediction_market
from fixtures import make_state, run_pipeline_metrics
from report.generate import _structured_data


def test_outcome_probability_in_report_matches_causal():
    state = run_pipeline_metrics(make_state())
    graph = causal.build(state)
    state.causal = graph
    data = _structured_data(state)
    assert data["outcome_probability"] == graph.overall_prediction


def test_causal_includes_prediction_market_when_sequential():
    state = run_pipeline_metrics(make_state())
    market = state.metrics["prediction_market"]["overall_outcome"]
    graph = causal.build(state)
    # Blended score should reflect market input (not ignore it entirely).
    assert abs(graph.overall_prediction - market) < 25.0


def test_refresh_changes_outcome_after_forecast():
    state = run_pipeline_metrics(make_state())
    graph_before = causal.build(state)
    state.causal = graph_before

    # Simulate forecast landing after initial causal (production ordering bug).
    from state import ForecastPoint, ForecastReadyPayload

    state.forecast = ForecastReadyPayload(
        model="test",
        metric="demand",
        horizon_days=90,
        mase_score=0.8,
        history=[ForecastPoint(date="2025-01-01", value=100.0)],
        predictions=[ForecastPoint(date="2025-04-01", value=130.0)],
        intervals=[],
    )
    refreshed = round(causal.compute_outcome_probability(state), 1)
    assert refreshed != graph_before.overall_prediction
