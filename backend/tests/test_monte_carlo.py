from __future__ import annotations

from agents import monte_carlo
from fixtures import make_state, run_pipeline_metrics


def test_monte_carlo_percentile_ordering():
    state = run_pipeline_metrics(make_state())
    metrics = monte_carlo.to_metrics(monte_carlo.run(state))
    p = metrics["outcome_percentiles"]

    assert p["p5"] <= p["p25"] <= p["p50"] <= p["p75"] <= p["p95"]
    assert metrics["n_simulations"] == 2000


def test_monte_carlo_scenario_keys():
    state = run_pipeline_metrics(make_state())
    metrics = monte_carlo.to_metrics(monte_carlo.run(state))

    for key in ("best_case", "worst_case", "most_likely", "black_swan"):
        assert "outcome" in metrics[key]
        assert "description" in metrics[key]

    assert metrics["best_case"]["outcome"] >= metrics["worst_case"]["outcome"]
