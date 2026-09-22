"""Monte Carlo scenario re-simulation wrapper."""
from __future__ import annotations

from typing import Any

import numpy as np

from causal_intel.models import CausalIntelGraph, ScenarioAssumption


def run_scenario_monte_carlo(
    graph: CausalIntelGraph,
    assumptions: list[ScenarioAssumption],
    n_sims: int = 500,
) -> dict[str, Any]:
    prior = graph.overall_prediction
    if graph.monte_carlo and graph.monte_carlo.get("outcome_percentiles"):
        prior = float(graph.monte_carlo["outcome_percentiles"].get("p50", prior))

    disabled = {a.node_id for a in assumptions if not a.enabled}
    overrides = {
        a.node_id: a.probability_override
        for a in assumptions
        if a.probability_override is not None
    }

    rng = np.random.default_rng(42)
    outcomes = np.empty(n_sims, dtype=float)
    for i in range(n_sims):
        o = prior
        for n in graph.nodes:
            if n.node_type == "goal" or n.id in disabled:
                continue
            p = overrides.get(n.id, n.probability)
            o += rng.normal((p - 50) * 0.08, 3.0)
        outcomes[i] = float(np.clip(o, 0, 100))

    return {
        "n_simulations": n_sims,
        "outcome_percentiles": {
            "p5": round(float(np.percentile(outcomes, 5)), 1),
            "p50": round(float(np.percentile(outcomes, 50)), 1),
            "p95": round(float(np.percentile(outcomes, 95)), 1),
        },
        "mean": round(float(np.mean(outcomes)), 1),
    }
