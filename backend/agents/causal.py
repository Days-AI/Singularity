"""Causal inference engine (spec section 9).

Builds an enriched goal-centric causal graph with prediction scores,
criticality, and influence labels for the dashboard CausalMapD3 panel.
"""
from __future__ import annotations

import logging

import numpy as np

from state import CausalEdge, CausalGraphPayload, CausalNode, SingularityState

logger = logging.getLogger("singularity.causal")

_SERIES_LEN = 60
_MAX_LAG = 6
_P_THRESHOLD = 0.10
_CLUSTER_NAMES = ["Skeptics", "Pragmatists", "Enthusiasts"]


def compute_outcome_probability(state: SingularityState) -> float:
    """Canonical headline outcome score (0–100) for dashboard, causal graph, and report."""
    return compute_outcome_with_breakdown(state)[0]


def compute_outcome_with_breakdown(state: SingularityState) -> tuple[float, dict[str, float]]:
    """Return (headline score, per-stream component scores on 0–100 scale)."""
    return _overall_prediction(state)


def build(state: SingularityState) -> CausalGraphPayload:
    series = _assemble_series(state)
    if len(series) < 2:
        return _empty_graph(state.query)

    names = list(series.keys())
    matrix = {k: _resample(np.asarray(v, dtype=float), _SERIES_LEN) for k, v in series.items()}

    raw_edges: list[CausalEdge] = []
    for cause in names:
        for effect in names:
            if cause == effect:
                continue
            if _kind_for(effect) == "cause":
                continue
            p_value, lag = _granger(matrix[cause], matrix[effect])
            weight = _hawkes_weight(matrix[cause], matrix[effect], lag)
            raw_edges.append(
                CausalEdge(
                    source=_nid(cause),
                    target=_nid(effect),
                    p_value=round(p_value, 4),
                    weight=round(weight, 3),
                    lag=max(1, lag),
                    influence=_influence_label(p_value, weight),
                )
            )

    edges = [e for e in raw_edges if e.p_value <= _P_THRESHOLD]
    if len(edges) < 3:
        edges = sorted(raw_edges, key=lambda e: (e.p_value, -e.weight))[: max(3, len(names))]
    edges = sorted(edges, key=lambda e: -e.weight)[:12]

    nodes = _enriched_nodes(names, matrix, state)
    goal_id = "goal_root"
    nodes.append(
        CausalNode(
            id=goal_id,
            label=_truncate(state.query, 48),
            kind="goal",
            prediction=0.0,
            criticality=100.0,
            description="Primary outcome probability derived from simulation convergence.",
        )
    )

    overall, breakdown = compute_outcome_with_breakdown(state)
    nodes[-1] = nodes[-1].model_copy(update={"prediction": round(overall, 1)})

    # Connect forecast / sentiment mediators to goal
    effect_ids = {n.id for n in nodes if n.kind in ("effect", "mediator")}
    for eid in effect_ids:
        if any(e.target == eid for e in edges):
            src_edge = next(e for e in edges if e.target == eid)
            edges.append(
                CausalEdge(
                    source=eid,
                    target=goal_id,
                    p_value=max(0.001, src_edge.p_value * 0.5),
                    weight=min(0.95, src_edge.weight + 0.1),
                    lag=1,
                    influence=_influence_label(src_edge.p_value * 0.5, src_edge.weight + 0.1),
                )
            )

    _apply_criticality(nodes, edges)
    return CausalGraphPayload(
        root_goal=state.query,
        root_description=(
            "Convergence of web evidence, OCEAN simulation, prediction market, "
            "and forecast toward the stated outcome."
        ),
        overall_prediction=round(overall, 1),
        outcome_breakdown=breakdown,
        nodes=nodes,
        edges=edges,
    )


def _empty_graph(query: str) -> CausalGraphPayload:
    return CausalGraphPayload(
        root_goal=query,
        root_description="Insufficient series data for causal inference.",
        overall_prediction=50.0,
        nodes=[],
        edges=[],
    )


def _enriched_nodes(
    names: list[str], matrix: dict[str, np.ndarray], state: SingularityState
) -> list[CausalNode]:
    nodes: list[CausalNode] = []
    for name in names:
        vals = matrix[name]
        # Recent level of the series. Series already on a 0-100 scale (sentiment,
        # interest indices) are used directly; out-of-range series (e.g. financial
        # price/volume) are mapped to their percentile-of-range so every node's
        # prediction is a comparable 0-100 score instead of saturating at 100.
        recent = float(np.mean(vals[-14:]))
        lo, hi = float(np.min(vals)), float(np.max(vals))
        if 0.0 <= recent <= 100.0 and hi <= 100.0:
            pred = recent
        elif hi > lo:
            pred = (recent - lo) / (hi - lo) * 100.0
        else:
            pred = 50.0
        pred = float(np.clip(pred, 0, 100))
        kind = _kind_for(name)
        desc = _node_description(name, kind)
        nodes.append(
            CausalNode(
                id=_nid(name),
                label=name,
                kind=kind,  # type: ignore[arg-type]
                prediction=round(pred, 1),
                criticality=0.0,
                description=desc,
            )
        )
    return nodes


def _node_description(name: str, kind: str) -> str:
    n = name.lower()
    if "sentiment" in n:
        return "IPIP-300 population mean sentiment from agent simulation."
    if "demand" in n or "adoption" in n:
        return "Derived outcome signal from sentiment and audience behavior drivers."
    if "search" in n or "interest" in n:
        return "Audience interest signal from web evidence."
    if kind == "cause":
        return "Exogenous contextual or audience indicator series."
    return "Causal pathway variable in the simulation graph."


_OVERALL_WEIGHTS = {
    "sentiment": 0.30,  # population attitude (primary behavioral signal)
    "adoption": 0.25,   # mean action-likelihood (behavioral intent)
    "market": 0.25,     # prediction-market aggregate (when available)
    "forecast": 0.15,   # projected market trajectory
    "evidence": 0.05,   # external evidence sentiment
}


def _agentic_population_signal(state: SingularityState) -> tuple[float | None, float | None]:
    """Sentiment and adoption on 0–100 from agentic population (opinions/deliberation)."""
    delib = state.metrics.get("deliberation", {})

    sent_score: float | None = None
    adopt_score: float | None = None

    if delib.get("mean_sentiment") is not None:
        sent_score = float(np.clip(50 + float(delib["mean_sentiment"]) * 50, 0, 100))
    elif state.persona_opinions:
        sent = float(np.mean([o.sentiment for o in state.persona_opinions]))
        sent_score = float(np.clip(50 + sent * 50, 0, 100))

    if delib.get("mean_action_likelihood") is not None:
        adopt_score = float(np.clip(float(delib["mean_action_likelihood"]) * 100, 0, 100))
    elif state.persona_opinions:
        actions = [o.action_likelihood for o in state.persona_opinions]
        if actions:
            adopt_score = float(np.clip(float(np.mean(actions)) * 100, 0, 100))

    return sent_score, adopt_score


def _overall_prediction(state: SingularityState) -> tuple[float, dict[str, float]]:
    """Precise, deterministic outcome probability in ``[0, 100]`` with stream breakdown."""
    components: list[tuple[str, float, float]] = []  # (name, score_0_100, weight)
    breakdown: dict[str, float] = {}

    market = state.metrics.get("prediction_market", {})
    if market.get("overall_outcome") is not None:
        m_score = float(np.clip(float(market["overall_outcome"]), 0, 100))
        components.append(("market", m_score, _OVERALL_WEIGHTS["market"]))
        breakdown["market"] = round(m_score, 1)

    agentic_sent, agentic_adopt = _agentic_population_signal(state)
    if agentic_sent is not None:
        components.append(("agentic_sentiment", agentic_sent, _OVERALL_WEIGHTS["sentiment"]))
        breakdown["agentic_sentiment"] = round(agentic_sent, 1)
    elif state.persona_responses:
        sent = float(np.mean([r.sentiment_score for r in state.persona_responses]))
        s_score = float(np.clip(50 + sent * 50, 0, 100))
        components.append(("agentic_sentiment", s_score, _OVERALL_WEIGHTS["sentiment"]))
        breakdown["agentic_sentiment"] = round(s_score, 1)

    if agentic_adopt is not None:
        components.append(("agentic_adoption", agentic_adopt, _OVERALL_WEIGHTS["adoption"]))
        breakdown["agentic_adoption"] = round(agentic_adopt, 1)
    elif state.persona_responses:
        actions = [r.action_likelihood for r in state.persona_responses if r.action_likelihood is not None]
        if actions:
            a_score = float(np.clip(float(np.mean(actions)) * 100, 0, 100))
            components.append(("agentic_adoption", a_score, _OVERALL_WEIGHTS["adoption"]))
            breakdown["agentic_adoption"] = round(a_score, 1)

    if state.forecast and state.forecast.predictions:
        hist = state.forecast.history
        start = hist[-1].value if hist else state.forecast.predictions[0].value
        end = state.forecast.predictions[-1].value
        pct_change = (end - start) / max(abs(start), 1e-9) * 100.0
        f_score = float(np.clip(50.0 + 50.0 * float(np.tanh(pct_change / 25.0)), 0, 100))
        components.append(("forecast", f_score, _OVERALL_WEIGHTS["forecast"]))
        breakdown["forecast"] = round(f_score, 1)

    evidence_sents = [e.sentiment for e in state.evidence if e.sentiment is not None]
    if evidence_sents:
        ev = float(np.mean(evidence_sents))
        e_score = float(np.clip(50 + ev * 50, 0, 100))
        components.append(("evidence", e_score, _OVERALL_WEIGHTS["evidence"]))
        breakdown["evidence"] = round(e_score, 1)

    if not components:
        return 50.0, breakdown
    total_weight = sum(w for _, _, w in components)
    blended = sum(score * w for _, score, w in components) / total_weight
    headline = float(np.clip(blended, 0, 100))
    breakdown["headline"] = round(headline, 1)
    return headline, breakdown


def _apply_criticality(nodes: list[CausalNode], edges: list[CausalEdge]) -> None:
    scores = {n.id: 0.0 for n in nodes}
    for e in edges:
        scores[e.source] = scores.get(e.source, 0) + e.weight
        scores[e.target] = scores.get(e.target, 0) + e.weight * 0.8
    max_s = max(scores.values()) if scores else 1.0
    for i, n in enumerate(nodes):
        if n.kind == "goal":
            continue
        crit = round((scores.get(n.id, 0) / max(max_s, 0.01)) * 100, 1)
        nodes[i] = n.model_copy(update={"criticality": crit})


def _influence_label(p_value: float, weight: float) -> str:
    if p_value < 0.02 and weight >= 0.6:
        return "++"
    if p_value < 0.05 or weight >= 0.5:
        return "+"
    if p_value > 0.08:
        return "-"
    return "+"


def _truncate(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


# --- series assembly ---------------------------------------------------------
def _assemble_series(state: SingularityState) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for ts in state.series:
        if len(ts.values) >= 8:
            out[ts.name] = ts.values
    rng = np.random.default_rng(0xCA5A1)

    base_sent = (
        float(np.mean([r.sentiment_score for r in state.persona_responses]))
        if state.persona_responses
        else 0.1
    )
    sentiment = _ar1_series(base_sent * 50 + 50, n=_SERIES_LEN, phi=0.7, noise=4.0, rng=rng)
    out["Agent Sentiment"] = sentiment.tolist()

    driver = None
    for name, vals in out.items():
        if _kind_for(name) == "cause":
            driver = _resample(np.asarray(vals, dtype=float), _SERIES_LEN)
            break
    demand = 0.6 * _znorm(sentiment) + (0.4 * _znorm(driver) if driver is not None else 0.0)
    demand = 100 + 12 * np.convolve(demand, np.ones(3) / 3, mode="same") + rng.normal(0, 1.5, _SERIES_LEN)
    out["Outcome Signal"] = demand.tolist()
    return out


def _kind_for(name: str) -> str:
    n = name.lower()
    if "sentiment" in n:
        return "mediator"
    if "outcome" in n or "demand" in n or "adoption" in n:
        return "effect"
    if "forecast" in n or "projection" in n:
        return "effect"
    return "cause"


# --- Granger -----------------------------------------------------------------
def _granger(cause: np.ndarray, effect: np.ndarray) -> tuple[float, int]:
    try:
        from statsmodels.tsa.stattools import grangercausalitytests

        data = np.column_stack([effect, cause])
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
    p = float(max(0.001, 1.0 - best_c))
    return p, best_lag


def _hawkes_weight(cause: np.ndarray, effect: np.ndarray, lag: int) -> float:
    branching = _hawkes_branching(_events_from(effect))
    xcorr = abs(_xcorr(cause, effect, max(1, lag)))
    return float(np.clip(0.5 * branching + 0.5 * xcorr, 0.0, 1.0))


def _events_from(series: np.ndarray) -> np.ndarray:
    diff = np.diff(series, prepend=series[0])
    thr = diff.mean() + 0.5 * (diff.std() or 1.0)
    idx = np.where(diff > thr)[0].astype(float)
    return idx


def _hawkes_branching(events: np.ndarray) -> float:
    if len(events) < 4:
        return 0.2
    T = float(events[-1] - events[0]) or 1.0
    events = events - events[0]
    try:
        from scipy.optimize import minimize

        def neg_ll(params: np.ndarray) -> float:
            mu, alpha, beta = np.exp(params)
            if beta <= 0:
                return 1e9
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
        res = minimize(neg_ll, x0, method="Nelder-Mead", options={"maxiter": 400, "xatol": 1e-3, "fatol": 1e-3})
        _, alpha, beta = np.exp(res.x)
        return float(np.clip(alpha / beta, 0.0, 0.99))
    except Exception as exc:  # noqa: BLE001
        logger.debug("hawkes fallback: %s", exc)
        gaps = np.diff(events)
        if len(gaps) == 0:
            return 0.2
        cv = gaps.std() / (gaps.mean() + 1e-9)
        return float(np.clip((cv - 1.0) * 0.5 + 0.3, 0.0, 0.95))


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
