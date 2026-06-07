"""Pydantic models for flow state and SSE event payloads.

These payload models are the Python mirror of frontend/src/types/events.ts and
are the single source of truth for what the backend serializes onto the wire.
Field names MUST match the TypeScript interfaces exactly.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

AgentType = Literal["web_search", "financial", "psychometric", "forecast"]
NodeStatus = Literal["pending", "running", "done", "failed"]


# --- DAG ---------------------------------------------------------------------
class DagNode(BaseModel):
    id: str
    task: str
    agent_type: AgentType
    dependencies: list[str] = Field(default_factory=list)
    priority: int = 1


class DagEdge(BaseModel):
    source: str
    target: str


class DagCreatedPayload(BaseModel):
    root_query: str
    nodes: list[DagNode]
    edges: list[DagEdge]


# --- Agents ------------------------------------------------------------------
class EvidenceItem(BaseModel):
    source: str
    title: str
    detail: str
    value: float | None = None
    unit: str | None = None
    url: str | None = None


class AgentStartedPayload(BaseModel):
    agent_id: str
    task: str
    agent_type: AgentType


class AgentResultPayload(BaseModel):
    agent_id: str
    agent_type: AgentType
    data: list[EvidenceItem]
    confidence: float
    duration_ms: int


# --- Psychometrics -----------------------------------------------------------
class OceanScores(BaseModel):
    O: float
    C: float
    E: float
    A: float
    N: float


class PersonaPoint(BaseModel):
    id: str
    pca: tuple[float, float, float]
    ocean: OceanScores
    sentiment: float
    cluster: int


class HeatmapRow(BaseModel):
    facet: str
    values: list[float]


class SentimentBucket(BaseModel):
    bucket: float
    count: int


class PersonaBatchPayload(BaseModel):
    batch_index: int
    batch_total: int
    profiles_in_batch: int
    cumulative_profiles: int
    ocean_mean: OceanScores
    sentiment_dist: list[SentimentBucket]
    points: list[PersonaPoint]
    heatmap: list[HeatmapRow]


# --- Forecast ----------------------------------------------------------------
class ForecastPoint(BaseModel):
    date: str
    value: float


class ForecastInterval(BaseModel):
    date: str
    lower: float
    upper: float


class ForecastReadyPayload(BaseModel):
    model: str
    metric: str
    horizon_days: int
    history: list[ForecastPoint]
    predictions: list[ForecastPoint]
    intervals: list[ForecastInterval]
    mase_score: float


# --- Causal ------------------------------------------------------------------
class CausalNode(BaseModel):
    id: str
    label: str
    kind: Literal["cause", "effect", "mediator"]


class CausalEdge(BaseModel):
    source: str
    target: str
    p_value: float
    weight: float
    lag: int


class CausalGraphPayload(BaseModel):
    nodes: list[CausalNode]
    edges: list[CausalEdge]


# --- Report / lifecycle ------------------------------------------------------
class ReportSectionPayload(BaseModel):
    index: int
    total: int
    section: str
    content: str
    status: Literal["streaming", "final"] = "final"


class CompletePayload(BaseModel):
    session_id: str
    duration_ms: int
    nodes_resolved: int


class ErrorPayload(BaseModel):
    code: str
    message: str
    node_id: str | None = None


# --- Flow state --------------------------------------------------------------
class PersonaResponse(BaseModel):
    """One archetype's simulated response (PT-02 output)."""

    archetype_id: str
    ocean: OceanScores
    facets: dict[str, float] = Field(default_factory=dict)
    sentiment_score: float = 0.0
    behavioral_intent: str = ""
    emotional_state: str = ""
    key_concerns: list[str] = Field(default_factory=list)
    purchase_likelihood: float = 0.0
    # OCEAN re-derived from the generated text by the personality engine.
    validated_ocean: OceanScores | None = None


class TimeSeries(BaseModel):
    """A named series the causal/forecast engines can consume."""

    name: str
    dates: list[str]
    values: list[float]


class SingularityState(BaseModel):
    """Typed state threaded through every flow node (spec section 3.2)."""

    query: str
    flow_uuid: str
    session_id: str | None = None
    dag: DagCreatedPayload | None = None
    evidence: list[EvidenceItem] = Field(default_factory=list)
    series: list[TimeSeries] = Field(default_factory=list)
    persona_responses: list[PersonaResponse] = Field(default_factory=list)
    ocean_mean: OceanScores | None = None
    forecast: ForecastReadyPayload | None = None
    causal: CausalGraphPayload | None = None
    report_sections: list[ReportSectionPayload] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
