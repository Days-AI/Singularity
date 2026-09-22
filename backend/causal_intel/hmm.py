"""Hidden Markov Model state tracking over forecast/history series."""
from __future__ import annotations

import numpy as np

from causal_intel.models import HMMResult, HMMState
from state import ForecastReadyPayload


def fit_hmm(forecast: ForecastReadyPayload | None, n_states: int = 4) -> HMMResult:
    if not forecast or not forecast.history:
        return HMMResult(
            states=[HMMState(name="Neutral", probability=1.0)],
            current_state="Neutral",
            transition_matrix=[[1.0]],
        )

    values = np.array([p.value for p in forecast.history[-30:]], dtype=float)
    if forecast.predictions:
        values = np.concatenate([values, [forecast.predictions[0].value]])

    if len(values) < 3:
        return HMMResult(
            states=[HMMState(name="Neutral", probability=1.0)],
            current_state="Neutral",
            transition_matrix=[[1.0]],
        )

    q = np.percentile(values, [25, 50, 75])
    state_names = ["Decline", "Caution", "Growth", "Surge"][:n_states]
    thresholds = [-np.inf, q[0], q[1], q[2], np.inf]

    def _state(v: float) -> int:
        for i in range(n_states):
            if thresholds[i] <= v < thresholds[i + 1]:
                return min(i, n_states - 1)
        return n_states - 1

    seq = [_state(v) for v in values]
    trans = np.zeros((n_states, n_states))
    for i in range(len(seq) - 1):
        trans[seq[i], seq[i + 1]] += 1
    row_sums = trans.sum(axis=1, keepdims=True)
    trans = np.where(row_sums > 0, trans / row_sums, 1.0 / n_states)

    counts = np.bincount(seq, minlength=n_states).astype(float)
    probs = counts / max(counts.sum(), 1)
    current = seq[-1]

    return HMMResult(
        states=[
            HMMState(name=state_names[i], probability=round(float(probs[i]), 3))
            for i in range(n_states)
        ],
        current_state=state_names[current],
        transition_matrix=trans.round(3).tolist(),
    )
