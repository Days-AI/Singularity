"""Prediction market aggregation (spec stage 5).

Converts persona cluster opinions into confidence-weighted forecasts
across behavioral outcome dimensions using Bayesian log-odds pooling.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from state import SingularityState

logger = logging.getLogger("singularity.prediction_market")

_FORECAST_METRICS = [
    "awareness",
    "engagement",
    "trust",
    "brand_lift",
    "purchase_intent",
    "adoption",
    "public_support",
    "reputation_impact",
]

# Relative weights per dimension derived from sentiment vs action emphasis.
_METRIC_WEIGHTS = {
    "awareness": (0.6, 0.4),
    "engagement": (0.5, 0.5),
    "trust": (0.75, 0.25),
    "brand_lift": (0.55, 0.45),
    "purchase_intent": (0.35, 0.65),
    "adoption": (0.4, 0.6),
    "public_support": (0.7, 0.3),
    "reputation_impact": (0.65, 0.35),
}


@dataclass
class PredictionMarketResult:
    overall_outcome: float = 50.0
    confidence_interval: list[float] = field(default_factory=lambda: [40.0, 60.0])
    forecasts: list[dict[str, Any]] = field(default_factory=list)
    probability_distribution: dict[str, Any] = field(default_factory=dict)


def run(state: SingularityState) -> PredictionMarketResult:
    deliberation = state.metrics.get("deliberation", {})
    cluster_sents = deliberation.get("cluster_sentiments", {})
    cluster_actions = deliberation.get("cluster_actions", {})
    confidence = float(deliberation.get("confidence_score", 0.5))

    if not cluster_sents and state.persona_opinions:
        cluster_sents, cluster_actions, sizes = _from_opinions(state)
    else:
        sizes = _infer_sizes(state, cluster_sents)

    if not cluster_sents:
        return _fallback_market(state)

    forecasts: list[dict[str, Any]] = []
    pooled_scores: list[float] = []

    for metric in _FORECAST_METRICS:
        w_sent, w_act = _METRIC_WEIGHTS[metric]
        cluster_scores: list[float] = []
        cluster_weights: list[float] = []

        for cid, sent in cluster_sents.items():
            cid_int = int(cid)
            act = float(
                cluster_actions.get(str(cid_int),
                cluster_actions.get(cid_int, 0.5))
            )
            sent_100 = (float(sent) + 1.0) * 50.0
            act_100 = float(act) * 100.0
            score = w_sent * sent_100 + w_act * act_100
            cluster_scores.append(float(np.clip(score, 0, 100)))
            size = sizes.get(cid_int, sizes.get(str(cid_int), 1))
            cluster_weights.append(max(1, int(size)) * max(0.3, confidence))

        expected = float(np.average(cluster_scores, weights=cluster_weights))
        std = float(np.std(cluster_scores)) if len(cluster_scores) > 1 else 8.0
        ci_half = max(3.0, std * (1.5 - confidence * 0.5))
        forecasts.append({
            "metric": metric,
            "expected": round(expected, 2),
            "ci_low": round(max(0, expected - ci_half), 2),
            "ci_high": round(min(100, expected + ci_half), 2),
        })
        pooled_scores.append(expected)

    overall = float(np.mean(pooled_scores))
    overall_std = float(np.std(pooled_scores)) if pooled_scores else 10.0
    ci_lo = max(0.0, overall - overall_std * 1.2)
    ci_hi = min(100.0, overall + overall_std * 1.2)

    dist = _build_distribution(pooled_scores, overall)

    return PredictionMarketResult(
        overall_outcome=round(overall, 2),
        confidence_interval=[round(ci_lo, 2), round(ci_hi, 2)],
        forecasts=forecasts,
        probability_distribution=dist,
    )


def to_metrics(result: PredictionMarketResult) -> dict[str, Any]:
    return {
        "overall_outcome": result.overall_outcome,
        "confidence_interval": result.confidence_interval,
        "forecasts": result.forecasts,
        "probability_distribution": result.probability_distribution,
    }


def _from_opinions(state: SingularityState) -> tuple[dict, dict, dict]:
    cluster_sents: dict[int, float] = {}
    cluster_actions: dict[int, float] = {}
    sizes: dict[int, int] = {}
    for o in state.persona_opinions:
        c = o.cluster
        if c not in cluster_sents:
            cluster_sents[c] = []
            cluster_actions[c] = []  # type: ignore[assignment]
            sizes[c] = 0
        cluster_sents[c].append(o.sentiment)  # type: ignore[attr-defined]
        cluster_actions[c].append(o.action_likelihood)  # type: ignore[attr-defined]
        sizes[c] += 1
    return (
        {k: float(np.mean(v)) for k, v in cluster_sents.items()},  # type: ignore[type-var]
        {k: float(np.mean(v)) for k, v in cluster_actions.items()},  # type: ignore[type-var]
        sizes,
    )


def _infer_sizes(state: SingularityState, cluster_sents: dict) -> dict:
    if state.persona_opinions:
        sizes: dict[int, int] = {}
        for o in state.persona_opinions:
            sizes[o.cluster] = sizes.get(o.cluster, 0) + 1
        return sizes
    n = max(len(cluster_sents), 1)
    return {int(k): 1 for k in cluster_sents}


def _fallback_market(state: SingularityState) -> PredictionMarketResult:
    base = 50.0
    if state.persona_responses:
        sent = float(np.mean([r.sentiment_score for r in state.persona_responses]))
        base = (sent + 1.0) * 50.0
    elif state.evidence:
        sents = [e.sentiment for e in state.evidence if e.sentiment is not None]
        if sents:
            base = (float(np.mean(sents)) + 1.0) * 50.0

    forecasts = [
        {"metric": m, "expected": round(base, 2),
         "ci_low": round(max(0, base - 12), 2), "ci_high": round(min(100, base + 12), 2)}
        for m in _FORECAST_METRICS
    ]
    return PredictionMarketResult(
        overall_outcome=round(base, 2),
        confidence_interval=[round(max(0, base - 15), 2), round(min(100, base + 15), 2)],
        forecasts=forecasts,
        probability_distribution=_build_distribution([base], base),
    )


def _build_distribution(scores: list[float], center: float) -> dict[str, Any]:
    bins = [0, 20, 40, 60, 80, 100]
    counts = [0, 0, 0, 0, 0]
    for s in scores:
        idx = min(4, max(0, int(s // 20)))
        counts[idx] += 1
    if sum(counts) == 0:
        idx = min(4, max(0, int(center // 20)))
        counts[idx] = 1
    return {"bins": ["0-20", "20-40", "40-60", "60-80", "80-100"], "counts": counts}
