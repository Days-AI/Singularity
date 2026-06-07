"""Track narrative adoption across social simulation rounds."""
from __future__ import annotations

from collections import Counter

import numpy as np

from agents.social.types import NarrativeSignal
from state import PersonaOpinion


def extract_narratives(opinions: list[PersonaOpinion], round_index: int) -> list[NarrativeSignal]:
    """Build narrative signals from winning deliberation voices and key concerns."""
    voice_counts: Counter[str] = Counter()
    concern_counts: Counter[str] = Counter()
    voice_sent: dict[str, list[float]] = {}

    for op in opinions:
        voice = _dominant_voice(op)
        voice_counts[voice] += 1
        voice_sent.setdefault(voice, []).append(op.sentiment)
        for c in op.key_concerns[:2]:
            concern_counts[c.lower()] += 1

    total = max(len(opinions), 1)
    signals: list[NarrativeSignal] = []

    for label, count in voice_counts.most_common(5):
        sents = voice_sent.get(label, [0.0])
        signals.append(
            NarrativeSignal(
                narrative_id=f"voice_{label.lower().replace(' ', '_')}",
                label=label,
                adoption_pct=round(count / total * 100, 1),
                sentiment=round(float(np.mean(sents)), 3),
                round_index=round_index,
            )
        )

    for concern, count in concern_counts.most_common(3):
        if count < 2:
            continue
        signals.append(
            NarrativeSignal(
                narrative_id=f"concern_{concern.replace(' ', '_')[:30]}",
                label=concern,
                adoption_pct=round(count / total * 100, 1),
                sentiment=round(
                    float(np.mean([o.sentiment for o in opinions if concern in " ".join(o.key_concerns).lower()])),
                    3,
                ),
                round_index=round_index,
            )
        )

    return signals[:8]


def _dominant_voice(op: PersonaOpinion) -> str:
    if op.active_biases:
        if "social_proof" in op.active_biases:
            return "Social Proof"
        if "loss_aversion" in op.active_biases:
            return "Risk Aversion"
    if op.sentiment > 0.25:
        return "Novelty Interest"
    if op.sentiment < -0.2:
        return "Price Sensitivity"
    return op.cluster_label or "Pragmatists"
