"""Domain-neutral causal intelligence graph models."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from state import CausalGraphPayload, EvidenceItem, ForecastReadyPayload

IntelNodeType = Literal["goal", "forecast", "event", "risk", "market", "driver"]
EdgePolarity = Literal["positive", "negative"]


class IntelNode(BaseModel):
    id: str
    label: str
    node_type: IntelNodeType
    question: str = ""
    description: str = ""
    probability: float = 50.0
    confidence: float = 0.5
    criticality: float = 0.0
    layer: int = 1
    group_id: str | None = None
    evidence_ids: list[int] = Field(default_factory=list)
    forecast_attached: bool = False
    trend: list[float] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IntelEdge(BaseModel):
    id: str
    source: str
    target: str
    polarity: EdgePolarity
    weight: float
    confidence: float = 0.5
    lag: int = 1
    flow_rate: float = 0.5
    label: str = "+"


class LinkedEvidence(BaseModel):
    index: int
    source: str
    title: str
    detail: str
    sentiment: float | None = None
    url: str | None = None


class EdgeSummary(BaseModel):
    id: str
    source_id: str
    source_label: str
    target_id: str
    target_label: str
    polarity: EdgePolarity
    weight: float
    label: str


class NodeDetail(BaseModel):
    node: IntelNode
    incoming: list[EdgeSummary]
    outgoing: list[EdgeSummary]
    evidence: list[LinkedEvidence]
    forecast: dict[str, Any] | None = None
    monte_carlo: dict[str, Any] | None = None
    paths_to_goal: list[list[str]] = Field(default_factory=list)
    explanation: str | None = None


class FlowNode(BaseModel):
    id: str
    type: str
    position: dict[str, float]
    data: dict[str, Any]


class FlowEdge(BaseModel):
    id: str
    source: str
    target: str
    type: str = "polarity"
    data: dict[str, Any] = Field(default_factory=dict)


class CausalIntelGraph(BaseModel):
    session_id: str | None = None
    flow_uuid: str | None = None
    query: str = ""
    root_goal: str = ""
    root_description: str = ""
    overall_prediction: float = 50.0
    nodes: list[IntelNode]
    edges: list[IntelEdge]
    flow_nodes: list[FlowNode] = Field(default_factory=list)
    flow_edges: list[FlowEdge] = Field(default_factory=list)
    monte_carlo: dict[str, Any] | None = None
    prediction_market: dict[str, Any] | None = None


class GraphSnapshot(BaseModel):
    model_config = ConfigDict(extra="ignore")

    query: str = ""
    causal: CausalGraphPayload | None = None
    evidence: list[EvidenceItem] = Field(default_factory=list)
    forecast: ForecastReadyPayload | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    deliberation: dict[str, Any] | None = None
    consensus: dict[str, Any] | None = None


class GraphRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    session_id: str | None = None
    flow_uuid: str | None = None
    query: str | None = None
    causal: CausalGraphPayload | None = None
    snapshot: GraphSnapshot | None = None


class ScenarioAssumption(BaseModel):
    node_id: str
    enabled: bool = True
    probability_override: float | None = None


class ScenarioRequest(BaseModel):
    session_id: str | None = None
    flow_uuid: str | None = None
    assumptions: list[ScenarioAssumption] = Field(default_factory=list)


class ScenarioResult(BaseModel):
    goal_probability: float
    baseline_goal: float
    delta: float
    node_deltas: dict[str, float] = Field(default_factory=dict)
    posterior_by_node: dict[str, float] = Field(default_factory=dict)


class InfluenceRank(BaseModel):
    node_id: str
    label: str
    score: float
    polarity: EdgePolarity | None = None


class SensitivityBar(BaseModel):
    node_id: str
    label: str
    low: float
    high: float
    baseline: float


class TimelineEvent(BaseModel):
    date: str
    label: str
    kind: str
    value: float | None = None


class AnalyticsBundle(BaseModel):
    influence_ranking: list[InfluenceRank]
    sensitivity: list[SensitivityBar]
    timeline: list[TimelineEvent]
    monte_carlo: dict[str, Any] | None = None


class HMMState(BaseModel):
    name: str
    probability: float


class HMMResult(BaseModel):
    states: list[HMMState]
    current_state: str
    transition_matrix: list[list[float]]


class PathDiscovery(BaseModel):
    paths: list[list[str]]
    labels: list[list[str]]
