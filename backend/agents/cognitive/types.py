"""Core types for the entropy-driven cognitive agent system."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from state import OceanScores


@dataclass
class ActiveBias:
    bias_id: str
    strength: float  # 0..1


@dataclass
class EmotionalState:
    trust: float = 0.5
    fear: float = 0.3
    curiosity: float = 0.5
    excitement: float = 0.4

    def as_dict(self) -> dict[str, float]:
        return {
            "trust": round(self.trust, 3),
            "fear": round(self.fear, 3),
            "curiosity": round(self.curiosity, 3),
            "excitement": round(self.excitement, 3),
        }

    def label(self) -> str:
        d = self.as_dict()
        dominant = max(d, key=d.get)  # type: ignore[arg-type]
        labels = {
            "trust": "trusting",
            "fear": "cautious",
            "curiosity": "curious",
            "excitement": "excited",
        }
        return labels.get(dominant, "measured")


@dataclass
class AgentMemory:
    past_campaigns: list[str] = field(default_factory=list)
    brand_experience: list[str] = field(default_factory=list)
    purchases: list[str] = field(default_factory=list)
    social_exposure: list[str] = field(default_factory=list)
    trust_modifier: float = 0.0  # -1..1 cumulative from experiences

    def has_negative_brand(self) -> bool:
        return any("negative" in x.lower() or "bad" in x.lower() for x in self.brand_experience)


@dataclass
class BeliefUtterance:
    node_id: str
    voice: str
    stance: float  # -1..1 contribution
    weight: float


@dataclass
class DeliberationTrace:
    utterances: list[BeliefUtterance] = field(default_factory=list)
    winning_voice: str = ""
    conflict_level: float = 0.0

    def summary_lines(self) -> list[str]:
        return [f'{u.voice}: "{u.stance:+.2f}" (w={u.weight:.2f})' for u in self.utterances[:6]]


@dataclass
class CognitiveStateVector:
    agent_id: str
    ocean: OceanScores
    facets: dict[str, float]
    active_biases: list[ActiveBias] = field(default_factory=list)
    emotions: EmotionalState = field(default_factory=EmotionalState)
    memory: AgentMemory = field(default_factory=AgentMemory)
    social_position: float = 0.3
    entropy_seed: int = 0
    total_entropy: float = 0.0
    confidence: float = 0.5
    uncertainty: float = 0.5
    deliberation: DeliberationTrace = field(default_factory=DeliberationTrace)
    social_influence_received: float = 0.0
    cluster: int = 0
    cluster_label: str = "Pragmatists"
    archetype_id: str = "arch_00"

    def facet_norm(self, name: str) -> float:
        return self.facets.get(name, 50.0) / 100.0

    def active_bias_ids(self) -> list[str]:
        return [b.bias_id for b in self.active_biases if b.strength > 0.15]


@dataclass
class AgentCognitiveOutput:
    """Deliberation output for one agent before NLG."""

    state: CognitiveStateVector
    sentiment: float
    behavioral_intent: str
    key_concerns: list[str]
    action_likelihood: float
    comment: str = ""
    response_source: str = "programmatic"

    def to_opinion_fields(self) -> dict[str, Any]:
        return {
            "sentiment": self.sentiment,
            "behavioral_intent": self.behavioral_intent,
            "emotional_state": self.state.emotions.label(),
            "key_concerns": self.key_concerns,
            "action_likelihood": self.action_likelihood,
            "comment": self.comment,
            "stance_confidence": round(self.state.confidence, 3),
            "uncertainty": round(self.state.uncertainty, 3),
            "active_biases": self.state.active_bias_ids(),
            "response_source": self.response_source,
            "facets": dict(self.state.facets),
        }
