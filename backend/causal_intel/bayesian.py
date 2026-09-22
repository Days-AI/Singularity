"""Bayesian log-odds updates for scenario assumptions."""
from __future__ import annotations

from causal_intel.models import ScenarioAssumption


def _to_logit(p: float) -> float:
    p = max(0.01, min(0.99, p / 100.0))
    import math
    return math.log(p / (1.0 - p))


def _from_logit(x: float) -> float:
    import math
    p = 1.0 / (1.0 + math.exp(-x))
    return max(0.0, min(100.0, p * 100.0))


def apply_bayesian_posterior(
    posteriors: dict[str, float],
    edges: list,
    assumptions: list[ScenarioAssumption],
    goal_id: str,
) -> dict[str, float]:
    """Lightweight log-odds nudge from toggled assumptions."""
    updated = dict(posteriors)
    override_map = {
        a.node_id: a.probability_override
        for a in assumptions
        if a.probability_override is not None
    }
    for nid, p in override_map.items():
        if nid in updated:
            prior = updated[nid]
            logit = _to_logit(prior) * 0.4 + _to_logit(p) * 0.6
            updated[nid] = _from_logit(logit)

    # Propagate one hop toward goal
    for e in edges:
        if e.target == goal_id and e.source in updated:
            src = updated[e.source]
            sign = 1 if e.polarity == "positive" else -1
            nudge = sign * (src - 50.0) * e.weight * 0.15
            goal = updated.get(goal_id, 50.0)
            updated[goal_id] = max(0.0, min(100.0, goal + nudge))

    return updated
