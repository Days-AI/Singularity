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
CausalNodeKind = Literal["cause", "effect", "mediator", "goal"]
InfluenceLabel = Literal["++", "+", "-", "--"]


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
    # Lexicon-derived polarity in [-1, 1]; drives evidence-feed row coloring.
    sentiment: float | None = None


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


class FacetScore(BaseModel):
    """One salient IPIP facet driving a persona's stance."""

    name: str
    score: float
    band: str  # "high" | "moderate" | "low"


class PersonaOpinion(BaseModel):
    id: str
    archetype_id: str
    cluster: int
    cluster_label: str
    ocean: OceanScores
    sentiment: float
    behavioral_intent: str
    emotional_state: str
    key_concerns: list[str]
    action_likelihood: float
    # First-person comment grounded in cognitive deliberation.
    comment: str = ""
    # Top facet drivers for this persona (highest absolute deviation from 50).
    top_facets: list[FacetScore] = Field(default_factory=list)
    # --- cognitive agent extensions (optional, backward-compatible) ---
    facets: dict[str, float] = Field(default_factory=dict)
    stance_confidence: float | None = None
    uncertainty: float | None = None
    active_biases: list[str] = Field(default_factory=list)
    response_source: Literal["programmatic", "llm"] | None = None


class NarrativeCluster(BaseModel):
    label: str
    size: int
    sentiment: float


class DeliberationPayload(BaseModel):
    agreement_rate: float
    polarization_index: float
    confidence_score: float
    narrative_clusters: list[NarrativeCluster] = Field(default_factory=list)
    cluster_sentiments: dict[str, float] = Field(default_factory=dict)
    cluster_actions: dict[str, float] = Field(default_factory=dict)
    persona_archetypes: list[str] = Field(default_factory=list)
    entropy_mean: float = 0.0
    social_contagion_index: float = 0.0


# --- Social interaction layer ------------------------------------------------
class SocialInteractionTickPayload(BaseModel):
    round: int
    debates: list[dict[str, Any]] = Field(default_factory=list)
    persuasion_events: list[dict[str, Any]] = Field(default_factory=list)
    narratives: list[dict[str, Any]] = Field(default_factory=list)
    polarization_index: float = 0.0
    mean_sentiment: float = 0.0


class SocialSimulationPayload(BaseModel):
    rounds_completed: int
    final_narratives: list[dict[str, Any]] = Field(default_factory=list)
    contagion_index: float = 0.0
    polarization_index: float = 0.0
    mean_sentiment: float = 0.0


# --- Specialist council ------------------------------------------------------
SpecialistId = Literal["pr", "brand", "marketing", "consumer_psychology"]


class CouncilOpinionPayload(BaseModel):
    specialist_id: SpecialistId
    role: str
    recommendation: str
    risks: list[str] = Field(default_factory=list)
    confidence: float = 0.5


class CouncilReadyPayload(BaseModel):
    opinions: list[CouncilOpinionPayload] = Field(default_factory=list)
    synthesis: str = ""


# --- Consensus engine --------------------------------------------------------
class ConsensusPayload(BaseModel):
    agreement_score: float
    recommended_action: str
    dissent: str = ""
    supporting_signals: list[str] = Field(default_factory=list)
    council_alignment: float = 0.0
    ranked_actions: list[str] = Field(default_factory=list)


class PersonaBatchPayload(BaseModel):
    batch_index: int
    batch_total: int
    profiles_in_batch: int
    cumulative_profiles: int
    ocean_mean: OceanScores
    points: list[PersonaPoint]
    heatmap: list[HeatmapRow]
    opinions: list[PersonaOpinion] = Field(default_factory=list)


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
    kind: CausalNodeKind
    prediction: float = 0.0
    criticality: float = 0.0
    description: str = ""


class CausalEdge(BaseModel):
    source: str
    target: str
    p_value: float
    weight: float
    lag: int
    influence: InfluenceLabel = "+"


class CausalGraphPayload(BaseModel):
    root_goal: str = ""
    root_description: str = ""
    overall_prediction: float = 0.0
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
    action_likelihood: float = 0.0
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
    # Master switch for live web sources (Serper/DDG/Wikipedia/yfinance).
    # When False, evidence agents fall back to deterministic synthetic data.
    web_sources_enabled: bool = True
    dag: DagCreatedPayload | None = None
    evidence: list[EvidenceItem] = Field(default_factory=list)
    series: list[TimeSeries] = Field(default_factory=list)
    persona_responses: list[PersonaResponse] = Field(default_factory=list)
    persona_opinions: list[PersonaOpinion] = Field(default_factory=list)
    ocean_mean: OceanScores | None = None
    forecast: ForecastReadyPayload | None = None
    causal: CausalGraphPayload | None = None
    report_sections: list[ReportSectionPayload] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
