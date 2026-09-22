"""FastAPI routes for causal intelligence."""
from __future__ import annotations

import uuid

import session_registry
from fastapi import APIRouter, Depends, HTTPException, Query

from auth import require_auth
from causal_intel.adapter import enrich_snapshot, snapshot_from_state
from causal_intel.engine import CausalIntelEngine
from causal_intel.explainer import explain_node
from causal_intel.hmm import fit_hmm
from causal_intel.models import (
    AnalyticsBundle,
    CausalIntelGraph,
    GraphRequest,
    GraphSnapshot,
    HMMResult,
    PathDiscovery,
    ScenarioRequest,
    ScenarioResult,
)
from causal_intel.simulation import run_scenario_monte_carlo
from causal_intel import store as graph_store

router = APIRouter()


def _coalesce_snapshot(body: GraphRequest) -> GraphSnapshot | None:
    if body.snapshot is not None:
        snap = body.snapshot
        if snap.causal is None and body.causal is not None:
            return snap.model_copy(update={"causal": body.causal, "query": snap.query or body.query or ""})
        return snap
    if body.causal is not None:
        return GraphSnapshot(query=body.query or "", causal=body.causal)
    return None


def _resolve_graph(body: GraphRequest) -> CausalIntelGraph:
    snapshot = _coalesce_snapshot(body)
    if snapshot and snapshot.causal and snapshot.causal.nodes:
        cache_id = body.session_id or body.flow_uuid or f"ephemeral-{uuid.uuid4().hex[:10]}"
        graph = enrich_snapshot(
            snapshot,
            session_id=cache_id,
            flow_uuid=body.flow_uuid,
        )
        graph_store.put(graph)
        return graph

    sid = body.session_id or body.flow_uuid
    if sid:
        cached = graph_store.get(session_id=body.session_id, flow_uuid=body.flow_uuid)
        if cached:
            return cached
        state = session_registry.get(sid)
        if state and state.causal and state.causal.nodes:
            snap = snapshot_from_state(state)
            graph = enrich_snapshot(
                snap,
                session_id=state.session_id or sid,
                flow_uuid=state.flow_uuid,
            )
            graph_store.put(graph)
            return graph

    raise HTTPException(
        status_code=404,
        detail=(
            "causal graph not available; POST a snapshot with causal.nodes, "
            "or session_id/flow_uuid after causal inference completes. "
            "If the route itself 404s with detail 'Not Found', restart the backend "
            "(stop.bat then run.bat) so /api/causal-intel is loaded."
        ),
    )


def _engine_for(sid: str | None, flow_uuid: str | None) -> CausalIntelEngine:
    graph = graph_store.get(session_id=sid, flow_uuid=flow_uuid)
    if not graph:
        raise HTTPException(status_code=404, detail="graph not cached; POST /graph first")
    evidence = []
    state = session_registry.get(sid or flow_uuid or "")
    if state:
        evidence = list(state.evidence)
    return CausalIntelEngine(graph, evidence)


@router.post("/graph", response_model=CausalIntelGraph)
async def post_graph(body: GraphRequest, _auth=Depends(require_auth)) -> CausalIntelGraph:
    return _resolve_graph(body)


@router.get("/graph/{session_id}", response_model=CausalIntelGraph)
async def get_graph(session_id: str, _auth=Depends(require_auth)) -> CausalIntelGraph:
    cached = graph_store.get(session_id=session_id)
    if cached:
        return cached
    state = session_registry.get(session_id)
    if not state or not state.causal:
        raise HTTPException(status_code=404, detail="session or causal data not found")
    graph = enrich_snapshot(
        snapshot_from_state(state),
        session_id=session_id,
        flow_uuid=state.flow_uuid,
    )
    graph_store.put(graph)
    return graph


@router.get("/nodes/{node_id}")
async def get_node_detail(
    node_id: str,
    session_id: str | None = Query(default=None),
    flow_uuid: str | None = Query(default=None),
    explain: bool = Query(default=False),
    _auth=Depends(require_auth),
):
    engine = _engine_for(session_id, flow_uuid)
    detail = engine.get_node_detail(node_id)
    if not detail:
        raise HTTPException(status_code=404, detail="node not found")
    if explain:
        detail.explanation = await explain_node(detail, engine.graph)
    return detail


@router.post("/scenario", response_model=ScenarioResult)
async def post_scenario(body: ScenarioRequest, _auth=Depends(require_auth)) -> ScenarioResult:
    engine = _engine_for(body.session_id, body.flow_uuid)
    return engine.propagate_scenario(body.assumptions)


@router.get("/analytics/{session_id}", response_model=AnalyticsBundle)
async def get_analytics(session_id: str, _auth=Depends(require_auth)) -> AnalyticsBundle:
    engine = _engine_for(session_id, None)
    return engine.analytics_bundle()


@router.post("/simulate/monte-carlo")
async def post_monte_carlo(body: ScenarioRequest, _auth=Depends(require_auth)) -> dict:
    engine = _engine_for(body.session_id, body.flow_uuid)
    return run_scenario_monte_carlo(engine.graph, body.assumptions)


@router.get("/hmm/{session_id}", response_model=HMMResult)
async def get_hmm(session_id: str, _auth=Depends(require_auth)) -> HMMResult:
    graph = graph_store.get(session_id=session_id)
    if not graph:
        state = session_registry.get(session_id)
        if state:
            graph = enrich_snapshot(snapshot_from_state(state), session_id=session_id)
    forecast = None
    state = session_registry.get(session_id)
    if state:
        forecast = state.forecast
    return fit_hmm(forecast)


@router.get("/paths/{session_id}", response_model=PathDiscovery)
async def get_paths(
    session_id: str,
    start_id: str = Query(...),
    end_id: str | None = Query(default=None),
    k: int = Query(default=5, ge=1, le=10),
    _auth=Depends(require_auth),
) -> PathDiscovery:
    engine = _engine_for(session_id, None)
    goal = end_id or engine.goal_id
    return engine.discover_paths(start_id, goal, k=k)
