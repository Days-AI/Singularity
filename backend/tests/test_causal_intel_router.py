"""HTTP tests for causal_intel graph endpoint."""
from __future__ import annotations

import session_registry
from agents import causal
from fastapi.testclient import TestClient
from fixtures import make_state, run_pipeline_metrics
from main import app


def test_post_graph_accepts_top_level_causal():
    state = run_pipeline_metrics(make_state())
    state.causal = causal.build(state)
    client = TestClient(app)
    r = client.post(
        "/api/causal-intel/graph",
        json={
            "session_id": "sess-1",
            "flow_uuid": "flow-1",
            "query": state.query,
            "causal": state.causal.model_dump(),
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["nodes"]
    assert body["session_id"] == "sess-1"


def test_post_graph_resolves_live_session_registry():
    state = run_pipeline_metrics(make_state())
    state.causal = causal.build(state)
    state.session_id = "live-session"
    state.flow_uuid = "live-flow"
    session_registry.touch(state)

    client = TestClient(app)
    r = client.post("/api/causal-intel/graph", json={"session_id": "live-session"})
    assert r.status_code == 200
    assert r.json()["nodes"]
