"""Monte Carlo scenario futures (spec stage 6).

Generates thousands of behavioral outcome paths with trend shifts, viral events,
competitive actions, and narrative perturbations.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from config import get_settings
from state import SingularityState

logger = logging.getLogger("singularity.monte_carlo")


@dataclass
class MonteCarloResult:
    outcome_percentiles: dict[str, float] = field(default_factory=dict)
    best_case: dict[str, Any] = field(default_factory=dict)
    worst_case: dict[str, Any] = field(default_factory=dict)
    most_likely: dict[str, Any] = field(default_factory=dict)
    black_swan: dict[str, Any] = field(default_factory=dict)
    n_simulations: int = 0


def run(state: SingularityState) -> MonteCarloResult:
    n_sims = get_settings().monte_carlo_simulations
    market = state.metrics.get("prediction_market", {})
    prior = float(market.get("overall_outcome", 50.0))
    deliberation = state.metrics.get("deliberation", {})
    polarization = float(deliberation.get("polarization_index", 0.2))
    confidence = float(deliberation.get("confidence_score", 0.5))

    evidence_sent = _evidence_sentiment(state)
    rng = np.random.default_rng(_seed(state.query))

    volatility = 8.0 + polarization * 20.0 + (1.0 - confidence) * 10.0
    outcomes = np.empty(n_sims, dtype=float)

    for i in range(n_sims):
        outcome = prior

        # Trend shift (sentiment drift)
        drift = rng.normal(evidence_sent * 5.0, volatility * 0.3)
        outcome += drift

        # Viral event (heavy tail, ~3% probability)
        if rng.random() < 0.03:
            outcome += rng.choice([-1, 1]) * rng.exponential(15.0)

        # Competitive action (~8% negative shock)
        if rng.random() < 0.08:
            outcome -= rng.uniform(5.0, 18.0)

        # Media reaction (evidence-weighted)
        if rng.random() < 0.12:
            outcome += evidence_sent * rng.uniform(3.0, 12.0)

        # Narrative/policy shift from population polarization
        if rng.random() < 0.10:
            outcome += rng.normal(0, polarization * 15.0)

        # Forecast trajectory influence if available
        if state.forecast and state.forecast.predictions:
            hist = state.forecast.history
            start = hist[-1].value if hist else state.forecast.predictions[0].value
            end = state.forecast.predictions[-1].value
            pct = (end - start) / max(abs(start), 1e-9) * 100.0
            outcome += float(np.tanh(pct / 30.0)) * rng.uniform(2.0, 8.0)

        outcomes[i] = float(np.clip(outcome, 0.0, 100.0))

    percentiles = {
        "p5": round(float(np.percentile(outcomes, 5)), 2),
        "p25": round(float(np.percentile(outcomes, 25)), 2),
        "p50": round(float(np.percentile(outcomes, 50)), 2),
        "p75": round(float(np.percentile(outcomes, 75)), 2),
        "p95": round(float(np.percentile(outcomes, 95)), 2),
    }

    p50 = percentiles["p50"]
    p95 = percentiles["p95"]
    p5 = percentiles["p5"]

    best_idx = int(np.argmax(outcomes))
    worst_idx = int(np.argmin(outcomes))
    best_val = float(outcomes[best_idx])
    worst_val = float(outcomes[worst_idx])

    black_swan_threshold = prior + 2.5 * volatility
    tail_mask = outcomes >= black_swan_threshold
    tail_prob = float(np.mean(tail_mask)) if n_sims else 0.0
    tail_outcome = float(np.max(outcomes[tail_mask])) if np.any(tail_mask) else p95

    return MonteCarloResult(
        outcome_percentiles=percentiles,
        best_case={
            "outcome": round(best_val, 2),
            "probability": round(float(np.mean(outcomes >= best_val - 1.0)), 4),
            "drivers": ["positive viral amplification", "strong segment convergence", "favorable media cycle"],
            "description": f"Upside scenario reaching {best_val:.0f}/100 with aligned narrative spread.",
        },
        worst_case={
            "outcome": round(worst_val, 2),
            "probability": round(float(np.mean(outcomes <= worst_val + 1.0)), 4),
            "drivers": ["competitive counter-narrative", "trust erosion", "polarized rejection"],
            "description": f"Downside scenario falling to {worst_val:.0f}/100 under sustained opposition.",
        },
        most_likely={
            "outcome": p50,
            "probability": round(float(np.mean(np.abs(outcomes - p50) <= 5.0)), 4),
            "drivers": ["gradual adoption curve", "mixed segment response", "moderate media attention"],
            "description": f"Central path at median {p50:.0f}/100 reflecting balanced population reaction.",
        },
        black_swan={
            "outcome": round(tail_outcome, 2),
            "probability": round(max(tail_prob, 0.005), 4),
            "trigger": "unexpected viral event or policy shift",
            "description": f"Low-probability tail event pushing outcome beyond {black_swan_threshold:.0f}/100.",
        },
        n_simulations=n_sims,
    )


def to_metrics(result: MonteCarloResult) -> dict[str, Any]:
    return {
        "outcome_percentiles": result.outcome_percentiles,
        "best_case": result.best_case,
        "worst_case": result.worst_case,
        "most_likely": result.most_likely,
        "black_swan": result.black_swan,
        "n_simulations": result.n_simulations,
    }


def _seed(query: str) -> int:
    return (sum(ord(c) for c in query) * 7919) % (2**31)


def _evidence_sentiment(state: SingularityState) -> float:
    sents = [e.sentiment for e in state.evidence if e.sentiment is not None]
    if sents:
        return float(np.mean(sents))
    if state.persona_responses:
        return float(np.mean([r.sentiment_score for r in state.persona_responses]))
    return 0.0
