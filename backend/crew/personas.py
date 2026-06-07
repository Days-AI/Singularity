"""Aggregate the 1,500-persona population into compact, injectable context.

CrewAI agents must never receive 1,500 raw rows. This module distills the
population into cluster-level OCEAN means, facet-driver frequencies, dominant
intents/emotions, and a handful of representative persona comments - small
enough to fit a prompt while preserving the behavioral signal.
"""
from __future__ import annotations

from collections import Counter

from config import get_settings
from state import PersonaOpinion, SingularityState

_OCEAN_DIMS = ("O", "C", "E", "A", "N")


def _mean(xs: list[float]) -> float:
    return round(sum(xs) / len(xs), 1) if xs else 0.0


def _ocean_mean(opinions: list[PersonaOpinion]) -> dict[str, float]:
    out: dict[str, float] = {}
    for dim in _OCEAN_DIMS:
        out[dim] = _mean([getattr(o.ocean, dim) for o in opinions])
    return out


def _mode(values: list[str]) -> str:
    values = [v for v in values if v]
    if not values:
        return ""
    return Counter(values).most_common(1)[0][0]


def build_persona_context(state: SingularityState) -> dict:
    """Distill state.persona_opinions into a compact, structured summary."""
    settings = get_settings()
    opinions = state.persona_opinions
    population = state.metrics.get("personas", len(opinions))
    if not opinions:
        return {"population": population, "clusters": [], "overall_ocean": None}

    by_cluster: dict[str, list[PersonaOpinion]] = {}
    for op in opinions:
        by_cluster.setdefault(op.cluster_label or "Unlabeled", []).append(op)

    clusters = []
    for label, members in sorted(by_cluster.items(), key=lambda kv: -len(kv[1])):
        # Facet-driver frequency across this cluster's top_facets.
        facet_counter: Counter = Counter()
        facet_scores: dict[str, list[float]] = {}
        for op in members:
            for f in op.top_facets:
                facet_counter[(f.name, f.band)] += 1
                facet_scores.setdefault(f.name, []).append(f.score)
        top_facets = [
            {
                "name": name,
                "band": band,
                "prevalence": round(count / len(members), 2),
                "mean_score": _mean(facet_scores.get(name, [])),
            }
            for (name, band), count in facet_counter.most_common(5)
        ]
        sample_comments = [
            op.comment for op in members if op.comment.strip()
        ][: settings.crew_max_personas_per_cluster]

        clusters.append(
            {
                "label": label,
                "size": len(members),
                "share": round(len(members) / len(opinions), 3),
                "ocean_mean": _ocean_mean(members),
                "mean_sentiment": _mean([o.sentiment for o in members]),
                "mean_action_likelihood": _mean([o.action_likelihood for o in members]),
                "dominant_intent": _mode([o.behavioral_intent for o in members]),
                "dominant_emotion": _mode([o.emotional_state for o in members]),
                "top_facet_drivers": top_facets,
                "representative_comments": sample_comments,
            }
        )

    return {
        "population": population,
        "overall_ocean": state.ocean_mean.model_dump() if state.ocean_mean else _ocean_mean(opinions),
        "clusters": clusters,
    }


def render_persona_context(ctx: dict) -> str:
    """Render the aggregated context as compact prose for prompt injection."""
    if not ctx or not ctx.get("clusters"):
        return "No persona population available."
    lines = [f"Population: {ctx.get('population')} simulated personas."]
    om = ctx.get("overall_ocean")
    if om:
        lines.append(
            "Overall OCEAN mean: "
            + " ".join(f"{d}{round(om[d])}" for d in _OCEAN_DIMS if d in om)
        )
    for c in ctx["clusters"]:
        om = c["ocean_mean"]
        ocean_str = " ".join(f"{d}{round(om[d])}" for d in _OCEAN_DIMS)
        drivers = ", ".join(
            f"{f['name']} ({f['band']}, {int(f['prevalence']*100)}%)"
            for f in c["top_facet_drivers"]
        )
        lines.append(
            f"\nCluster '{c['label']}' - {c['size']} personas ({int(c['share']*100)}%):"
        )
        lines.append(f"  OCEAN {ocean_str}; sentiment {c['mean_sentiment']:+.2f}; "
                     f"action-likelihood {c['mean_action_likelihood']:.2f}")
        lines.append(f"  Intent: {c['dominant_intent']}; Emotion: {c['dominant_emotion']}")
        if drivers:
            lines.append(f"  Facet drivers: {drivers}")
        for cm in c["representative_comments"]:
            lines.append(f"  \"{cm}\"")
    return "\n".join(lines)
