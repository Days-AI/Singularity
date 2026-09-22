"""Adapter tests for causal_intel."""
from __future__ import annotations

from agents import causal, prediction_market
from causal_intel.adapter import enrich_snapshot, snapshot_from_state
from causal_intel.engine import CausalIntelEngine
from fixtures import make_state, run_pipeline_metrics


def test_adapter_produces_intel_nodes():
    state = run_pipeline_metrics(make_state())
    graph = causal.build(state)
    state.causal = graph
    snap = snapshot_from_state(state)
    intel = enrich_snapshot(snap, session_id="test-session")

    assert len(intel.nodes) == len(graph.nodes)
    assert len(intel.edges) == len(graph.edges)
    assert intel.overall_prediction == graph.overall_prediction
    assert intel.flow_nodes
    assert intel.flow_edges

    goal = next(n for n in intel.nodes if n.node_type == "goal")
    assert goal.layer == 0
    assert goal.probability == graph.overall_prediction


def test_adapter_polarity_on_edges():
    state = run_pipeline_metrics(make_state())
    state.causal = causal.build(state)
    intel = enrich_snapshot(snapshot_from_state(state))
    for e in intel.edges:
        assert e.polarity in ("positive", "negative")
        assert e.label in ("+", "-")


def test_engine_node_detail():
    state = run_pipeline_metrics(make_state())
    state.causal = causal.build(state)
    intel = enrich_snapshot(snapshot_from_state(state))
    engine = CausalIntelEngine(intel, state.evidence)
    goal = next(n for n in intel.nodes if n.node_type == "goal")
    detail = engine.get_node_detail(goal.id)
    assert detail is not None
    assert detail.node.id == goal.id
