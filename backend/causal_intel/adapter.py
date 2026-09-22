"""Maps existing SingularityState / snapshot slices → CausalIntelGraph."""
from __future__ import annotations

import re
from typing import Any

from causal_intel.models import (
    CausalIntelGraph,
    EdgePolarity,
    FlowEdge,
    FlowNode,
    GraphSnapshot,
    IntelEdge,
    IntelNode,
    IntelNodeType,
)
from state import CausalGraphPayload, CausalNodeKind, EvidenceItem, ForecastReadyPayload

_LAYER: dict[CausalNodeKind, int] = {
    "goal": 0,
    "effect": 1,
    "mediator": 1,
    "cause": 2,
}

_TYPE: dict[CausalNodeKind, IntelNodeType] = {
    "goal": "goal",
    "cause": "risk",
    "effect": "market",
    "mediator": "driver",
}


def _polarity(weight: float, influence: str) -> EdgePolarity:
    if influence in ("--", "-"):
        return "negative"
    if influence in ("++", "+"):
        return "positive"
    return "positive" if weight >= 0.5 else "negative"


def _edge_label(polarity: EdgePolarity) -> str:
    return "+" if polarity == "positive" else "-"


def _confidence(p_value: float) -> float:
    return round(max(0.05, min(0.99, 1.0 - p_value)), 3)


def _match_evidence(label: str, evidence: list[EvidenceItem]) -> list[int]:
    tokens = set(re.findall(r"[a-z]{4,}", label.lower()))
    if not tokens:
        return []
    matched: list[int] = []
    for i, e in enumerate(evidence):
        hay = f"{e.title} {e.detail} {e.source}".lower()
        if any(t in hay for t in tokens):
            matched.append(i)
    return matched[:5]


def _trend_from_forecast(
    forecast: ForecastReadyPayload | None, label: str
) -> list[float]:
    if not forecast:
        return []
    metric = forecast.metric.lower()
    if metric not in label.lower() and "forecast" not in label.lower():
        return []
    hist = [p.value for p in forecast.history[-12:]]
    pred = [p.value for p in forecast.predictions[:6]]
    return hist + pred


def _question_for(label: str, kind: CausalNodeKind, query: str) -> str:
    if kind == "goal":
        return query or label
    return f"How does {label.lower()} influence the outcome?"


def enrich_snapshot(
    snapshot: GraphSnapshot,
    session_id: str | None = None,
    flow_uuid: str | None = None,
) -> CausalIntelGraph:
    causal = snapshot.causal
    if not causal or not causal.nodes:
        return CausalIntelGraph(
            session_id=session_id,
            flow_uuid=flow_uuid,
            query=snapshot.query,
            root_goal=snapshot.query,
            nodes=[],
            edges=[],
        )

    evidence = snapshot.evidence
    forecast = snapshot.forecast
    metrics = snapshot.metrics or {}
    mc = metrics.get("monte_carlo")
    pm = metrics.get("prediction_market")

    intel_nodes: list[IntelNode] = []
    for n in causal.nodes:
        ntype = _TYPE.get(n.kind, "driver")
        layer = _LAYER.get(n.kind, 1)
        trend = _trend_from_forecast(forecast, n.label)
        intel_nodes.append(
            IntelNode(
                id=n.id,
                label=n.label,
                node_type=ntype,
                question=_question_for(n.label, n.kind, snapshot.query),
                description=n.description or "",
                probability=round(n.prediction, 1),
                confidence=0.85 if n.kind == "goal" else 0.55,
                criticality=round(n.criticality, 1),
                layer=layer,
                group_id=ntype if n.kind != "goal" else None,
                evidence_ids=_match_evidence(n.label, evidence),
                forecast_attached=bool(trend),
                trend=trend,
                metadata={"kind": n.kind},
            )
        )

    intel_edges: list[IntelEdge] = []
    for i, e in enumerate(causal.edges):
        pol = _polarity(e.weight, e.influence)
        conf = _confidence(e.p_value)
        intel_edges.append(
            IntelEdge(
                id=f"e_{i}_{e.source}_{e.target}",
                source=e.source,
                target=e.target,
                polarity=pol,
                weight=round(e.weight, 3),
                confidence=conf,
                lag=e.lag,
                flow_rate=round(e.weight * conf, 3),
                label=_edge_label(pol),
            )
        )

    graph = CausalIntelGraph(
        session_id=session_id,
        flow_uuid=flow_uuid,
        query=snapshot.query,
        root_goal=causal.root_goal or snapshot.query,
        root_description=causal.root_description,
        overall_prediction=round(causal.overall_prediction, 1),
        nodes=intel_nodes,
        edges=intel_edges,
        monte_carlo=mc if isinstance(mc, dict) else None,
        prediction_market=pm if isinstance(pm, dict) else None,
    )
    graph.flow_nodes, graph.flow_edges = _to_flow(graph)
    return graph


def _to_flow(graph: CausalIntelGraph) -> tuple[list[FlowNode], list[FlowEdge]]:
    layer_y = {0: 40.0, 1: 220.0, 2: 400.0}
    by_layer: dict[int, list[IntelNode]] = {}
    for n in graph.nodes:
        by_layer.setdefault(n.layer, []).append(n)

    flow_nodes: list[FlowNode] = []
    for layer, nodes in by_layer.items():
        span = max(len(nodes), 1)
        for i, n in enumerate(nodes):
            x = 80.0 + (520.0 / max(span - 1, 1)) * i if span > 1 else 300.0
            y = layer_y.get(layer, 220.0)
            w = 200 if n.node_type == "goal" else 168
            h = 88 if n.node_type == "goal" else 76
            flow_nodes.append(
                FlowNode(
                    id=n.id,
                    type="goalCard" if n.node_type == "goal" else "causalCard",
                    position={"x": x, "y": y},
                    data={
                        "label": n.label,
                        "question": n.question,
                        "probability": n.probability,
                        "confidence": n.confidence,
                        "criticality": n.criticality,
                        "nodeType": n.node_type,
                        "trend": n.trend,
                        "groupId": n.group_id,
                    },
                )
            )
            # store dimensions for layout reflow
            n.metadata["width"] = w
            n.metadata["height"] = h

    flow_edges = [
        FlowEdge(
            id=e.id,
            source=e.source,
            target=e.target,
            type="polarity",
            data={
                "polarity": e.polarity,
                "weight": e.weight,
                "label": e.label,
                "flowRate": e.flow_rate,
            },
        )
        for e in graph.edges
    ]
    return flow_nodes, flow_edges


def snapshot_from_state(state: Any) -> GraphSnapshot:
    """Build GraphSnapshot from SingularityState."""
    return GraphSnapshot(
        query=state.query,
        causal=state.causal,
        evidence=list(state.evidence),
        forecast=state.forecast,
        metrics=dict(state.metrics),
        deliberation=state.metrics.get("deliberation"),
        consensus=state.metrics.get("consensus"),
    )
