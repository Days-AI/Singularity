"""Population-level deliberation metrics for downstream agents."""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from agents.cognitive.types import AgentCognitiveOutput


def aggregate_deliberation_from_opinions(
    opinions: list[Any],
    social_contagion_index: float = 0.0,
    entropy_mean: float = 0.0,
) -> dict[str, Any]:
    """Rebuild deliberation metrics from PersonaOpinion rows after social simulation."""
    if not opinions:
        return _empty_deliberation()

    sentiments = np.array([float(o.sentiment) for o in opinions])
    actions = np.array([float(o.action_likelihood) for o in opinions])
    confidences = np.array([
        float(o.stance_confidence if o.stance_confidence is not None else 0.5)
        for o in opinions
    ])

    mean_sent = float(sentiments.mean())
    std_sent = float(sentiments.std()) if len(sentiments) > 1 else 0.0
    polarization = min(1.0, std_sent * 2.5)
    agreement = float(np.mean(np.abs(sentiments - mean_sent) < 0.25))

    cluster_sents: dict[str, float] = {}
    cluster_actions: dict[str, float] = {}
    for label in ("Skeptics", "Pragmatists", "Enthusiasts"):
        mask = [o for o in opinions if o.cluster_label == label]
        if mask:
            cluster_sents[label] = round(float(np.mean([m.sentiment for m in mask])), 3)
            cluster_actions[label] = round(float(np.mean([m.action_likelihood for m in mask])), 3)

    groups: dict[str, list[float]] = {}
    for o in opinions:
        voice = o.cluster_label or "Mixed"
        if o.key_concerns:
            voice = o.key_concerns[0]
        groups.setdefault(voice, []).append(float(o.sentiment))
    narrative_clusters = [
        {"label": k, "size": len(v), "sentiment": round(float(np.mean(v)), 3)}
        for k, v in sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)
    ][:8]

    arch_ids = [o.archetype_id for o in opinions]
    counts = Counter(arch_ids).most_common(3)
    persona_archetypes = [
        f"{arch_id} ({count / max(len(opinions), 1):.0%} of population)"
        for arch_id, count in counts
    ]

    return {
        "agreement_rate": round(agreement, 3),
        "polarization_index": round(polarization, 3),
        "confidence_score": round(float(confidences.mean()), 3),
        "narrative_clusters": narrative_clusters,
        "persona_archetypes": persona_archetypes,
        "cluster_sentiments": cluster_sents,
        "cluster_actions": cluster_actions,
        "entropy_mean": round(entropy_mean, 4),
        "social_contagion_index": round(social_contagion_index, 4),
        "mean_sentiment": round(mean_sent, 3),
        "mean_action_likelihood": round(float(actions.mean()), 3),
    }


def aggregate_deliberation_metrics(
    outputs: list[AgentCognitiveOutput],
    responses_archetypes: list[Any],
    social_contagion_index: float = 0.0,
    entropy_mean: float = 0.0,
) -> dict[str, Any]:
    """Build state.metrics['deliberation'] consumed by prediction_market, etc."""
    if not outputs:
        return _empty_deliberation()

    sentiments = np.array([o.sentiment for o in outputs])
    actions = np.array([o.action_likelihood for o in outputs])
    confidences = np.array([o.state.confidence for o in outputs])

    mean_sent = float(sentiments.mean())
    std_sent = float(sentiments.std()) if len(sentiments) > 1 else 0.0
    polarization = min(1.0, std_sent * 2.5)

    # Agreement: share within 0.25 of mean sentiment
    agreement = float(np.mean(np.abs(sentiments - mean_sent) < 0.25))

    cluster_sents: dict[str, float] = {}
    cluster_actions: dict[str, float] = {}
    for label in ("Skeptics", "Pragmatists", "Enthusiasts"):
        mask = [o for o in outputs if o.state.cluster_label == label]
        if mask:
            cluster_sents[label] = round(float(np.mean([m.sentiment for m in mask])), 3)
            cluster_actions[label] = round(float(np.mean([m.action_likelihood for m in mask])), 3)

    narrative_clusters = _narrative_clusters(outputs)
    persona_archetypes = _top_archetypes(outputs, responses_archetypes)

    return {
        "agreement_rate": round(agreement, 3),
        "polarization_index": round(polarization, 3),
        "confidence_score": round(float(confidences.mean()), 3),
        "narrative_clusters": narrative_clusters,
        "persona_archetypes": persona_archetypes,
        "cluster_sentiments": cluster_sents,
        "cluster_actions": cluster_actions,
        "entropy_mean": round(entropy_mean, 4),
        "social_contagion_index": round(social_contagion_index, 4),
        "mean_sentiment": round(mean_sent, 3),
        "mean_action_likelihood": round(float(actions.mean()), 3),
    }


def _narrative_clusters(outputs: list[AgentCognitiveOutput]) -> list[dict[str, Any]]:
    """Group by dominant winning deliberation voice."""
    groups: dict[str, list[float]] = {}
    for o in outputs:
        voice = o.state.deliberation.winning_voice or "Mixed"
        groups.setdefault(voice, []).append(o.sentiment)

    clusters: list[dict[str, Any]] = []
    for voice, sents in sorted(groups.items(), key=lambda x: len(x[1]), reverse=True):
        clusters.append({
            "label": voice,
            "size": len(sents),
            "sentiment": round(float(np.mean(sents)), 3),
        })
    return clusters[:8]


def _top_archetypes(outputs: list[AgentCognitiveOutput], responses: list[Any]) -> list[str]:
    arch_ids = [o.state.archetype_id for o in outputs]
    counts = Counter(arch_ids).most_common(3)
    labels: list[str] = []
    for arch_id, count in counts:
        share = count / max(len(outputs), 1)
        labels.append(f"{arch_id} ({share:.0%} of population)")
    return labels


def _empty_deliberation() -> dict[str, Any]:
    return {
        "agreement_rate": 0.5,
        "polarization_index": 0.2,
        "confidence_score": 0.5,
        "narrative_clusters": [],
        "persona_archetypes": [],
        "cluster_sentiments": {},
        "cluster_actions": {},
        "entropy_mean": 0.0,
        "social_contagion_index": 0.0,
    }
