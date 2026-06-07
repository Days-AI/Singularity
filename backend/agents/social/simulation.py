"""Multi-round social interaction orchestrator."""
from __future__ import annotations

import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from agents.cognitive.aggregate import aggregate_deliberation_from_opinions
from agents.social.narrative import extract_narratives
from agents.social.types import DebateExchange, PersuasionAttempt, SocialRoundResult
from config import get_settings
from state import PersonaOpinion, SingularityState, SocialInteractionTickPayload, SocialSimulationPayload


@dataclass
class SocialSimulationResult:
    rounds: list[SocialRoundResult] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    final_payload: SocialSimulationPayload | None = None


def _cluster_means(opinions: list[PersonaOpinion]) -> dict[str, float]:
    groups: dict[str, list[float]] = {}
    for op in opinions:
        groups.setdefault(op.cluster_label, []).append(op.sentiment)
    return {k: float(np.mean(v)) for k, v in groups.items() if v}


def _polarization(opinions: list[PersonaOpinion]) -> float:
    if len(opinions) < 2:
        return 0.0
    sents = np.array([o.sentiment for o in opinions])
    return round(min(1.0, float(sents.std()) * 2.5), 3)


def _run_debates(opinions: list[PersonaOpinion], rng: random.Random) -> list[DebateExchange]:
    means = _cluster_means(opinions)
    labels = list(means.keys())
    debates: list[DebateExchange] = []
    if len(labels) < 2:
        return debates
    pairs = [(labels[i], labels[j]) for i in range(len(labels)) for j in range(i + 1, len(labels))]
    rng.shuffle(pairs)
    for a, b in pairs[:3]:
        sa, sb = means[a], means[b]
        debates.append(
            DebateExchange(
                cluster_a=a,
                cluster_b=b,
                topic="campaign stance",
                stance_a=round(sa, 3),
                stance_b=round(sb, 3),
                intensity=round(abs(sa - sb), 3),
            )
        )
    return debates


def _identify_influencers(opinions: list[PersonaOpinion], max_n: int = 8) -> list[PersonaOpinion]:
    scored = []
    for op in opinions:
        social_score = op.ocean.E / 100.0 * 0.5 + (op.facets.get("Assertiveness", 50) / 100.0) * 0.5
        if social_score >= 0.55:
            scored.append((social_score * abs(op.sentiment), op))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [op for _, op in scored[:max_n]]


def _apply_persuasion(
    opinions: list[PersonaOpinion],
    influencers: list[PersonaOpinion],
    rng: random.Random,
    strength: float = 0.12,
) -> list[PersuasionAttempt]:
    events: list[PersuasionAttempt] = []
    if not influencers:
        return events
    by_cluster: dict[str, list[PersonaOpinion]] = {}
    for op in opinions:
        by_cluster.setdefault(op.cluster_label, []).append(op)

    for inf in influencers:
        cluster = inf.cluster_label
        targets = by_cluster.get(cluster, [])
        if inf.sentiment <= 0:
            continue
        persuaded = 0
        for op in targets:
            if op.id == inf.id:
                continue
            agreeableness = op.ocean.A / 100.0
            social_proof = 1.0 if "social_proof" in (op.active_biases or []) else 0.5
            if rng.random() > agreeableness * social_proof * 0.7:
                continue
            delta = inf.sentiment * strength * rng.uniform(0.6, 1.0)
            op.sentiment = round(min(1.0, max(-1.0, op.sentiment + delta)), 3)
            op.action_likelihood = round(min(1.0, max(0.0, op.action_likelihood + delta * 0.4)), 3)
            persuaded += 1
        if persuaded:
            events.append(
                PersuasionAttempt(
                    influencer_id=inf.id,
                    target_cluster=cluster,
                    delta_sentiment=round(inf.sentiment * strength, 3),
                    success_rate=round(persuaded / max(len(targets) - 1, 1), 3),
                )
            )
    return events


def _run_round(opinions: list[PersonaOpinion], round_index: int, seed: int) -> SocialRoundResult:
    rng = random.Random(seed + round_index * 997)
    debates = _run_debates(opinions, rng)
    influencers = _identify_influencers(opinions)
    persuasion = _apply_persuasion(opinions, influencers, rng)
    narratives = extract_narratives(opinions, round_index)
    pol = _polarization(opinions)
    mean_s = float(np.mean([o.sentiment for o in opinions])) if opinions else 0.0

    # Debate cross-cluster pull: minor sentiment drift toward cluster mean blend
    means = _cluster_means(opinions)
    global_mean = float(np.mean(list(means.values()))) if means else 0.0
    for op in opinions:
        pull = (global_mean - op.sentiment) * 0.04 * rng.uniform(0.5, 1.0)
        op.sentiment = round(min(1.0, max(-1.0, op.sentiment + pull)), 3)

    return SocialRoundResult(
        round_index=round_index,
        debates=debates,
        persuasion_events=persuasion,
        narratives=narratives,
        polarization_index=pol,
        mean_sentiment=round(mean_s, 3),
    )


def _round_to_tick(r: SocialRoundResult) -> SocialInteractionTickPayload:
    return SocialInteractionTickPayload(
        round=r.round_index,
        debates=[
            {
                "cluster_a": d.cluster_a,
                "cluster_b": d.cluster_b,
                "topic": d.topic,
                "stance_a": d.stance_a,
                "stance_b": d.stance_b,
                "intensity": d.intensity,
            }
            for d in r.debates
        ],
        persuasion_events=[
            {
                "influencer_id": p.influencer_id,
                "target_cluster": p.target_cluster,
                "delta_sentiment": p.delta_sentiment,
                "success_rate": p.success_rate,
            }
            for p in r.persuasion_events
        ],
        narratives=[
            {
                "narrative_id": n.narrative_id,
                "label": n.label,
                "adoption_pct": n.adoption_pct,
                "sentiment": n.sentiment,
            }
            for n in r.narratives
        ],
        polarization_index=r.polarization_index,
        mean_sentiment=r.mean_sentiment,
    )


async def run(
    state: SingularityState,
    emit_tick: Callable[[SocialInteractionTickPayload], Awaitable[None]] | None = None,
) -> SocialSimulationResult:
    settings = get_settings()
    opinions = state.persona_opinions
    if not opinions:
        return SocialSimulationResult()

    n_rounds = max(1, settings.social_simulation_rounds)
    seed = hash(state.flow_uuid) & 0x7FFFFFFF
    rounds: list[SocialRoundResult] = []

    for r in range(n_rounds):
        round_result = _run_round(opinions, r + 1, seed)
        rounds.append(round_result)
        tick = _round_to_tick(round_result)
        if emit_tick:
            await emit_tick(tick)

    final_narratives = rounds[-1].narratives if rounds else []
    contagion = sum(len(r.persuasion_events) for r in rounds) / max(len(opinions), 1)
    contagion = round(min(1.0, contagion * 0.5), 4)

    final_payload = SocialSimulationPayload(
        rounds_completed=len(rounds),
        final_narratives=[
            {
                "narrative_id": n.narrative_id,
                "label": n.label,
                "adoption_pct": n.adoption_pct,
                "sentiment": n.sentiment,
            }
            for n in final_narratives
        ],
        contagion_index=contagion,
        polarization_index=rounds[-1].polarization_index if rounds else 0.0,
        mean_sentiment=rounds[-1].mean_sentiment if rounds else 0.0,
    )

    deliberation = aggregate_deliberation_from_opinions(
        opinions,
        social_contagion_index=contagion,
    )

    metrics = {
        "rounds_completed": len(rounds),
        "contagion_index": contagion,
        "polarization_index": final_payload.polarization_index,
        "mean_sentiment": final_payload.mean_sentiment,
        "final_narratives": final_payload.final_narratives,
        "deliberation_refresh": deliberation,
    }

    return SocialSimulationResult(rounds=rounds, metrics=metrics, final_payload=final_payload)
