"""Consensus engine — merge social, council, and analytics signals."""
from __future__ import annotations

from typing import Any

import numpy as np

from state import ConsensusPayload, SingularityState


def run(state: SingularityState) -> tuple[dict[str, Any], ConsensusPayload]:
    delib = state.metrics.get("deliberation", {})
    social = state.metrics.get("social_simulation", {})
    council = state.metrics.get("council", {})
    market = state.metrics.get("prediction_market", {})
    mc = state.metrics.get("monte_carlo", {})
    swarm = state.metrics.get("swarm_optimization", {})

    pop_agreement = float(delib.get("agreement_rate", 0.5))
    pop_polarization = float(delib.get("polarization_index", 0.2))
    market_outcome = float(market.get("overall_outcome", 50.0)) / 100.0
    mc_p50 = float(mc.get("outcome_percentiles", {}).get("p50", 50.0)) / 100.0
    contagion = float(social.get("contagion_index", 0.0))

    council_ops = council.get("opinions", [])
    council_conf = (
        float(np.mean([o.get("confidence", 0.5) for o in council_ops]))
        if council_ops
        else 0.5
    )

    agreement_score = round(
        0.35 * pop_agreement + 0.25 * council_conf + 0.25 * market_outcome + 0.15 * (1 - pop_polarization),
        3,
    )

    council_alignment = _council_alignment(council_ops, delib)
    recommended = _recommended_action(state, council, swarm, market_outcome)
    dissent = _dissent_note(pop_polarization, council_ops, delib)
    signals = _supporting_signals(state)
    ranked = _ranked_actions(state, recommended)

    payload = ConsensusPayload(
        agreement_score=agreement_score,
        recommended_action=recommended,
        dissent=dissent,
        supporting_signals=signals,
        council_alignment=round(council_alignment, 3),
        ranked_actions=ranked,
    )

    metrics = {
        "agreement_score": payload.agreement_score,
        "recommended_action": payload.recommended_action,
        "dissent": payload.dissent,
        "supporting_signals": payload.supporting_signals,
        "council_alignment": payload.council_alignment,
        "ranked_actions": payload.ranked_actions,
        "inputs": {
            "population_agreement": pop_agreement,
            "polarization": pop_polarization,
            "market_outcome": market_outcome,
            "monte_carlo_p50": mc_p50,
            "social_contagion": contagion,
            "council_confidence": council_conf,
        },
    }
    return metrics, payload


def _council_alignment(council_ops: list[dict], delib: dict) -> float:
    if not council_ops:
        return 0.5
    avg_conf = float(np.mean([o.get("confidence", 0.5) for o in council_ops]))
    pop_sent = float(delib.get("mean_sentiment", 0.0))
    positive_council = sum(1 for o in council_ops if "proceed" in o.get("recommendation", "").lower()
                          or "launch" in o.get("recommendation", "").lower()
                          or "invest" in o.get("recommendation", "").lower())
    align = 0.5 * avg_conf + 0.3 * (1 if pop_sent > 0 else 0.5) + 0.2 * (positive_council / len(council_ops))
    return min(1.0, max(0.0, align))


def _recommended_action(state: SingularityState, council: dict, swarm: dict, market: float) -> str:
    synthesis = council.get("synthesis", "").strip()
    if synthesis:
        return synthesis[:600]
    path = swarm.get("optimal_path", {}).get("steps", [])
    if path:
        return f"Execute swarm-optimized path: {' → '.join(str(s) for s in path[:4])}"
    if market > 0.6:
        return "Proceed with campaign rollout targeting receptive clusters first."
    if market < 0.4:
        return "Pause broad rollout; address trust and value concerns raised by Skeptics."
    return f"Run phased pilot for: {state.query[:120]}"


def _dissent_note(polarization: float, council_ops: list[dict], delib: dict) -> str:
    parts: list[str] = []
    if polarization > 0.35:
        parts.append("Population remains polarized after social simulation.")
    clusters = delib.get("narrative_clusters", [])
    if len(clusters) >= 3:
        parts.append(f"{len(clusters)} competing narratives persist.")
    low_conf = [o for o in council_ops if o.get("confidence", 1) < 0.45]
    if low_conf:
        parts.append(f"{len(low_conf)} council specialist(s) express low confidence.")
    return " ".join(parts) if parts else "Minor dissent; overall alignment acceptable."


def _supporting_signals(state: SingularityState) -> list[str]:
    signals: list[str] = []
    delib = state.metrics.get("deliberation", {})
    if delib.get("mean_sentiment") is not None:
        signals.append(f"Mean population sentiment {delib['mean_sentiment']:+.2f}")
    social = state.metrics.get("social_simulation", {})
    if social.get("contagion_index"):
        signals.append(f"Social contagion index {social['contagion_index']:.2f}")
    market = state.metrics.get("prediction_market", {})
    if market.get("overall_outcome") is not None:
        signals.append(f"Prediction market outcome {market['overall_outcome']:.0f}/100")
    if state.causal and state.causal.nodes:
        top = max(state.causal.nodes, key=lambda n: n.criticality)
        signals.append(f"Top causal driver: {top.label}")
    return signals[:6]


def _ranked_actions(state: SingularityState, primary: str) -> list[str]:
    actions = [primary[:200]]
    de = state.metrics.get("decision_engine", {})
    for opt in de.get("options", [])[:4]:
        action = opt.get("action", "")
        if action and action not in actions:
            actions.append(action[:200])
    return actions[:5]
