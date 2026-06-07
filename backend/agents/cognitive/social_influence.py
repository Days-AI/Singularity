"""Social influence: influencers shift follower adoption."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agents.cognitive.types import AgentCognitiveOutput, CognitiveStateVector


@dataclass
class InfluencerRecord:
    agent_index: int
    support: float  # -1..1 stance
    reach: float


def compute_social_position(ocean_row: np.ndarray, facets_row: np.ndarray, facet_names: list[str]) -> float:
    """0=follower .. 1=influencer from E + Assertiveness + Gregariousness."""
    e = float(ocean_row[2]) / 100.0
    idx_assert = facet_names.index("Assertiveness") if "Assertiveness" in facet_names else -1
    idx_greg = facet_names.index("Gregariousness") if "Gregariousness" in facet_names else -1
    assertiveness = float(facets_row[idx_assert]) / 100.0 if idx_assert >= 0 else 0.5
    greg = float(facets_row[idx_greg]) / 100.0 if idx_greg >= 0 else 0.5
    return min(1.0, 0.4 * e + 0.35 * assertiveness + 0.25 * greg)


def identify_influencers(
    outputs: list[AgentCognitiveOutput],
    pop_ocean: np.ndarray,
    pop_facets: np.ndarray,
    facet_names: list[str],
    max_influencers: int = 10,
) -> list[InfluencerRecord]:
    scores: list[tuple[int, float, float]] = []
    for i, out in enumerate(outputs):
        pos = out.state.social_position
        if pos < 0.55:
            continue
        scores.append((i, out.sentiment, pos))
    scores.sort(key=lambda x: x[2] * abs(x[1]), reverse=True)
    return [
        InfluencerRecord(agent_index=idx, support=sup, reach=pop_ocean[idx, 2] / 100.0)
        for idx, sup, _ in scores[:max_influencers]
    ]


def apply_social_influence(
    outputs: list[AgentCognitiveOutput],
    visual_clusters: np.ndarray,
    influencers: list[InfluencerRecord],
    contagion_strength: float = 0.22,
) -> float:
    """
    Pass 2: followers in same cluster as influencers shift adoption.
    Returns social_contagion_index (0..1).
    """
    if not influencers:
        return 0.0

    affected = 0
    for inf in influencers:
        cluster = int(visual_clusters[inf.agent_index])
        if inf.support <= 0:
            continue
        for i, out in enumerate(outputs):
            if i == inf.agent_index:
                continue
            if int(visual_clusters[i]) != cluster:
                continue
            if out.state.social_position > 0.6:
                continue
            delta = inf.support * contagion_strength * inf.reach
            out.sentiment = round(min(1.0, max(-1.0, out.sentiment + delta)), 3)
            out.action_likelihood = round(min(1.0, max(0.0, out.action_likelihood + delta * 0.5)), 3)
            out.state.social_influence_received += delta
            out.state.confidence = min(0.95, out.state.confidence + abs(delta) * 0.15)
            affected += 1

    return round(min(1.0, affected / max(len(outputs), 1)), 4)


def refresh_action_from_sentiment(output: AgentCognitiveOutput) -> None:
    base = 0.5 + output.sentiment * 0.35
    base += (50.0 - output.state.ocean.N) / 400.0
    base += output.state.social_influence_received * 0.2
    output.action_likelihood = round(max(0.0, min(1.0, base)), 3)
