"""Causal inference engine (spec section 9).

Two complementary signals per directed edge:
  - Granger causality p-value (statsmodels) -> does the cause's past improve
    prediction of the effect? (edge significance)
  - Hawkes self/cross-excitation via a custom exponential-kernel MLE (numpy +
    scipy.optimize) -> strength of event cascade (edge weight 0..1)

Runs on real evidence series where available (yfinance / pytrends), augmented
with a population-sentiment series and a derived demand series so the graph is
always populated. Everything degrades to correlation-based estimates if
statsmodels/scipy are missing.
"""
from __future__ import annotations

import logging

import numpy as np

from state import CausalEdge, CausalGraphPayload, CausalNode, SingularityState

logger = logging.getLogger("singularity.causal")

_SERIES_LEN = 60      # common resampled length
_MAX_LAG = 6
_P_THRESHOLD = 0.10


def build(state: SingularityState) -> CausalGraphPayload:
    series = _assemble_series(state)
    if len(series) < 2:
        return CausalGraphPayload(nodes=[], edges=[])

    names = list(series.keys())
    matrix = {k: _resample(np.asarray(v, dtype=float), _SERIES_LEN) for k, v in series.items()}

    nodes = [CausalNode(id=_nid(n), label=n, kind=_kind_for(n)) for n in names]

    raw_edges: list[CausalEdge] = []
    for cause in names:
        for effect in names:
            if cause == effect:
                continue
            if _kind_for(effect) == "cause":
                continue  # causes are exogenous sinks-of-causation
            p_value, lag = _granger(matrix[cause], matrix[effect])
            weight = _hawkes_weight(matrix[cause], matrix[effect], lag)
            raw_edges.append(
                CausalEdge(source=_nid(cause), target=_nid(effect),
                           p_value=round(p_value, 4), weight=round(weight, 3), lag=max(1, lag))
            )

    edges = [e for e in raw_edges if e.p_value <= _P_THRESHOLD]
    if len(edges) < 3:  # ensure a legible graph; keep strongest by weight
        edges = sorted(raw_edges, key=lambda e: (e.p_value, -e.weight))[:max(3, len(names))]
    edges = sorted(edges, key=lambda e: -e.weight)[:12]
    return CausalGraphPayload(nodes=nodes, edges=edges)


# --- series assembly ---------------------------------------------------------
def _assemble_series(state: SingularityState) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for ts in state.series:
        if len(ts.values) >= 8:
            out[ts.name] = ts.values
    rng = np.random.default_rng(0xCA5A1)

    base_sent = float(np.mean([r.sentiment_score for r in state.persona_responses])) \
        if state.persona_responses else 0.1
    sentiment = _ar1_series(base_sent * 50 + 50, n=_SERIES_LEN, phi=0.7, noise=4.0, rng=rng)
    out["Consumer Sentiment"] = sentiment.tolist()

    # Demand is driven by sentiment (lag) + the first real cause series if present.
    driver = None
    for name, vals in out.items():
        if _kind_for(name) == "cause":
            driver = _resample(np.asarray(vals, dtype=float), _SERIES_LEN)
            break
    demand = 0.6 * _znorm(sentiment) + (0.4 * _znorm(driver) if driver is not None else 0.0)
    demand = 100 + 12 * np.convolve(demand, np.ones(3) / 3, mode="same") + rng.normal(0, 1.5, _SERIES_LEN)
    out["Adoption Demand"] = demand.tolist()
    return out


def _kind_for(name: str) -> str:
    n = name.lower()
    if "sentiment" in n:
        return "mediator"
    if "demand" in n or "adoption" in n:
        return "effect"
    return "cause"


# --- Granger -----------------------------------------------------------------
def _granger(cause: np.ndarray, effect: np.ndarray) -> tuple[float, int]:
    try:
        from statsmodels.tsa.stattools import grangercausalitytests

        data = np.column_stack([effect, cause])  # [effect, cause] per statsmodels
        results = grangercausalitytests(data, maxlag=_MAX_LAG)
        best_p, best_lag = 1.0, 1
        for lag, res in results.items():
            p = float(res[0]["ssr_ftest"][1])
            if p < best_p:
                best_p, best_lag = p, lag
        return best_p, best_lag
    except Exception as exc:  # noqa: BLE001
        logger.debug("granger fallback: %s", exc)
        return _granger_fallback(cause, effect)


def _granger_fallback(cause: np.ndarray, effect: np.ndarray) -> tuple[float, int]:
    best_c, best_lag = 0.0, 1
    for lag in range(1, _MAX_LAG + 1):
        c = abs(_xcorr(cause, effect, lag))
        if c > best_c:
            best_c, best_lag = c, lag
    # Map correlation magnitude to a pseudo p-value (higher corr -> lower p).
    p = float(max(0.001, 1.0 - best_c))
    return p, best_lag


# --- Hawkes (custom exponential-kernel MLE) ----------------------------------
def _hawkes_weight(cause: np.ndarray, effect: np.ndarray, lag: int) -> float:
    branching = _hawkes_branching(_events_from(effect))
    xcorr = abs(_xcorr(cause, effect, max(1, lag)))
    return float(np.clip(0.5 * branching + 0.5 * xcorr, 0.0, 1.0))


def _events_from(series: np.ndarray) -> np.ndarray:
    """Shock timestamps: indices where the series jumps above mean + 0.5 std."""
    diff = np.diff(series, prepend=series[0])
    thr = diff.mean() + 0.5 * (diff.std() or 1.0)
    idx = np.where(diff > thr)[0].astype(float)
    return idx


def _hawkes_branching(events: np.ndarray) -> float:
    """Branching ratio alpha/beta from an exp-kernel Hawkes MLE.

    Intensity: lambda(t) = mu + sum_{t_i < t} alpha * exp(-beta (t - t_i)).
    Returns alpha/beta in [0,1) - the expected offspring per event.
    """
    if len(events) < 4:
        return 0.2
    T = float(events[-1] - events[0]) or 1.0
    events = events - events[0]
    try:
        from scipy.optimize import minimize

        def neg_ll(params: np.ndarray) -> float:
            mu, alpha, beta = np.exp(params)  # positivity via log-params
            if beta <= 0:
                return 1e9
            # Recursive computation of the exp-kernel sum.
            r = np.zeros(len(events))
            for i in range(1, len(events)):
                dt = events[i] - events[i - 1]
                r[i] = np.exp(-beta * dt) * (1.0 + r[i - 1])
            intensity = mu + alpha * r
            intensity = np.clip(intensity, 1e-9, None)
            term_sum = np.sum(np.log(intensity))
            compensator = mu * T + (alpha / beta) * np.sum(1.0 - np.exp(-beta * (T - events)))
            return float(compensator - term_sum)

        x0 = np.log(np.array([max(1e-3, len(events) / T), 0.5, 1.0]))
        res = minimize(neg_ll, x0, method="Nelder-Mead",
                       options={"maxiter": 400, "xatol": 1e-3, "fatol": 1e-3})
        _, alpha, beta = np.exp(res.x)
        return float(np.clip(alpha / beta, 0.0, 0.99))
    except Exception as exc:  # noqa: BLE001
        logger.debug("hawkes fallback: %s", exc)
        # Fallback: clustering of events as a crude branching proxy.
        gaps = np.diff(events)
        if len(gaps) == 0:
            return 0.2
        cv = gaps.std() / (gaps.mean() + 1e-9)  # >1 => bursty => more excitation
        return float(np.clip((cv - 1.0) * 0.5 + 0.3, 0.0, 0.95))


# --- numeric helpers ---------------------------------------------------------
def _resample(values: np.ndarray, n: int) -> np.ndarray:
    if len(values) == n:
        return values.astype(float)
    xp = np.linspace(0, 1, len(values))
    x = np.linspace(0, 1, n)
    return np.interp(x, xp, values).astype(float)


def _znorm(arr: np.ndarray) -> np.ndarray:
    s = arr.std() or 1.0
    return (arr - arr.mean()) / s


def _ar1_series(level: float, n: int, phi: float, noise: float, rng) -> np.ndarray:
    out = np.empty(n)
    out[0] = level
    for t in range(1, n):
        out[t] = level + phi * (out[t - 1] - level) + rng.normal(0, noise)
    return np.clip(out, 0, 100)


def _xcorr(cause: np.ndarray, effect: np.ndarray, lag: int) -> float:
    if lag >= len(cause):
        return 0.0
    c = _znorm(cause[: len(cause) - lag])
    e = _znorm(effect[lag:])
    if len(c) < 2:
        return 0.0
    return float(np.mean(c * e))


def _nid(name: str) -> str:
    return name.lower().replace(" ", "_").replace("/", "_").replace(".", "_")[:40]
