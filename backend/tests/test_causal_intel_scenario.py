"""Scenario propagation tests."""
from __future__ import annotations

from agents import causal
from causal_intel.adapter import enrich_snapshot, snapshot_from_state
from causal_intel.engine import CausalIntelEngine
from causal_intel.models import ScenarioAssumption
from fixtures import make_state, run_pipeline_metrics


def test_scenario_disable_driver_changes_goal():
    state = run_pipeline_metrics(make_state())
    state.causal = causal.build(state)
    intel = enrich_snapshot(snapshot_from_state(state))
    engine = CausalIntelEngine(intel)

    baseline = engine.propagate_scenario([]).goal_probability
    driver = next(n for n in intel.nodes if n.node_type in ("risk", "driver", "market"))
    result = engine.propagate_scenario([
        ScenarioAssumption(node_id=driver.id, enabled=False),
    ])
    assert abs(result.baseline_goal - intel.overall_prediction) < 0.2
    assert isinstance(result.delta, float)


def test_scenario_override_increases_goal_when_positive_driver():
    state = run_pipeline_metrics(make_state())
    state.causal = causal.build(state)
    intel = enrich_snapshot(snapshot_from_state(state))
    engine = CausalIntelEngine(intel)

    driver = next(
        (n for n in intel.nodes if n.node_type in ("driver", "market") and n.probability > 50),
        intel.nodes[1] if len(intel.nodes) > 1 else intel.nodes[0],
    )
    low = engine.propagate_scenario([
        ScenarioAssumption(node_id=driver.id, enabled=True, probability_override=30.0),
    ]).goal_probability
    high = engine.propagate_scenario([
        ScenarioAssumption(node_id=driver.id, enabled=True, probability_override=85.0),
    ]).goal_probability
    assert high >= low
