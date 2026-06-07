"""Multi-round social interaction simulation types."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DebateExchange:
    cluster_a: str
    cluster_b: str
    topic: str
    stance_a: float
    stance_b: float
    intensity: float


@dataclass
class PersuasionAttempt:
    influencer_id: str
    target_cluster: str
    delta_sentiment: float
    success_rate: float


@dataclass
class NarrativeSignal:
    narrative_id: str
    label: str
    adoption_pct: float
    sentiment: float
    round_index: int


@dataclass
class SocialRoundResult:
    round_index: int
    debates: list[DebateExchange] = field(default_factory=list)
    persuasion_events: list[PersuasionAttempt] = field(default_factory=list)
    narratives: list[NarrativeSignal] = field(default_factory=list)
    polarization_index: float = 0.0
    mean_sentiment: float = 0.0
