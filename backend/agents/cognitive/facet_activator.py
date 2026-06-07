"""Map query + evidence context to stimulus-relevant facet weights."""
from __future__ import annotations

import re

from agents import personality
from state import EvidenceItem

_FACET_ORDER: list[str] = [f for facets in personality.FACETS.values() for f in facets]

# Keywords that boost relevance of facet groups for the current stimulus.
_STIMULUS_FACET_HINTS: dict[str, list[str]] = {
    r"price|cost|expensive|cheap|budget|afford": ["Cautiousness", "Achievement", "Self-Discipline"],
    r"trust|brand|reliable|credible|reputation": ["Trust", "Morality", "Dutifulness"],
    r"new|novel|innov|genai|ai|tech|digital": ["Intellect", "Adventurousness", "Imagination", "Liberalism"],
    r"social|friend|community|people|peer": ["Gregariousness", "Friendliness", "Cooperation"],
    r"buy|purchase|shop|product|market": ["Achievement", "Cautiousness", "Immoderation"],
    r"risk|safe|uncertain|worry|fear": ["Anxiety", "Vulnerability", "Cautiousness"],
    r"policy|government|regul|public": ["Liberalism", "Morality", "Trust"],
    r"emotion|feel|sentiment|experience": ["Emotionality", "Sympathy", "Cheerfulness"],
}


def facet_activation_weights(
    query: str,
    evidence: list[EvidenceItem],
    facets: dict[str, float],
) -> dict[str, float]:
    """Return per-facet relevance multipliers (0.5..1.5) for deliberation weighting."""
    text = query.lower()
    for item in evidence[:5]:
        text += " " + (item.title or "").lower() + " " + (item.detail or "").lower()

    weights = {name: 1.0 for name in _FACET_ORDER}
    for pattern, facet_names in _STIMULUS_FACET_HINTS.items():
        if re.search(pattern, text):
            for fname in facet_names:
                if fname in weights:
                    weights[fname] = min(1.5, weights[fname] + 0.25)

    # Baseline: extreme facets are always somewhat relevant.
    for name, score in facets.items():
        if abs(score - 50.0) >= 20:
            weights[name] = min(1.5, weights.get(name, 1.0) + 0.1)

    return weights
