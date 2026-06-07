"""Forecasting engine (spec section 8).

Optional tier 1: Chronos (if installed) for native probabilistic intervals.
Primary path: Google TimesFM + Meta Prophet ensemble when both are available;
single-engine fallback otherwise. The `model` field always reflects the engine
actually used. Final safety net: damped-trend analytic projection.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import logging
from typing import Any

import numpy as np

from config import get_settings
from state import ForecastInterval, ForecastPoint, ForecastReadyPayload, SingularityState, TimeSeries

logger = logging.getLogger("singularity.forecast")

_HISTORY_TAIL = 60
_TIMESFM_MODEL: Any | None = None
_CHRONOS_PIPE: Any | None = None


async def run(state: SingularityState) -> ForecastReadyPayload:
    series = _select_target(state)
    horizon = get_settings().forecast_horizon_days
    return await asyncio.to_thread(_forecast_blocking, series, horizon)


def _select_target(state: SingularityState) -> TimeSeries:
    """Prefer the richest real series; otherwise synthesize a demand proxy."""
    candidates = [s for s in state.series if len(s.values) >= 12]
    if candidates:
        return max(candidates, key=lambda s: len(s.values))
    base = 100.0
    if state.persona_responses:
        base = 100.0 + 40.0 * float(np.mean([r.sentiment_score for r in state.persona_responses]))
    rng = np.random.default_rng(0xF0CA57)
    n = 120
    vals = [base]
    for i in range(1, n):
        vals.append(max(1.0, vals[-1] + 0.2 + 4 * np.sin(i / 14) * 0.1 + rng.normal(0, 1.2)))
    today = dt.date.today()
    dates = [(today - dt.timedelta(days=(n - i))).strftime("%Y-%m-%d") for i in range(n)]
    return TimeSeries(name="Projected Demand Index", dates=dates, values=vals)


def _forecast_blocking(series: TimeSeries, horizon: int) -> ForecastReadyPayload:
    values = np.asarray(series.values, dtype=float)
    dates = series.dates if len(series.dates) == len(values) else _synth_dates(len(values))

    chronos = _try_chronos(values, horizon)
    if chronos is not None:
        point, lower, upper, model_name = chronos
    else:
        point, lower, upper, model_name = _ensemble_or_single_forecast(values, dates, horizon)

    mase = _mase(values, dates, model_name)
    future_dates = _future_dates(dates[-1], horizon)
    history = [ForecastPoint(date=d, value=round(float(v), 3))
               for d, v in zip(dates[-_HISTORY_TAIL:], values[-_HISTORY_TAIL:])]
    predictions = [ForecastPoint(date=d, value=round(float(v), 3))
                   for d, v in zip(future_dates, point)]
    intervals = [ForecastInterval(date=d, lower=round(float(lo), 3), upper=round(float(hi), 3))
                 for d, lo, hi in zip(future_dates, lower, upper)]
    return ForecastReadyPayload(
        model=model_name, metric=series.name, horizon_days=horizon,
        history=history, predictions=predictions, intervals=intervals,
        mase_score=round(float(mase), 3),
    )


# --- optional Chronos --------------------------------------------------------
def _try_chronos(values: np.ndarray, horizon: int):
    global _CHRONOS_PIPE
    try:
        import torch  # noqa: F401
        from chronos import ChronosPipeline

        if _CHRONOS_PIPE is None:
            _CHRONOS_PIPE = ChronosPipeline.from_pretrained("amazon/chronos-t5-small")
        import torch as _torch

        context = _torch.tensor(values, dtype=_torch.float32)
        forecast = _CHRONOS_PIPE.predict(context, horizon)
        samples = forecast[0].numpy()
        point = np.quantile(samples, 0.5, axis=0)
        lower = np.quantile(samples, 0.05, axis=0)
        upper = np.quantile(samples, 0.95, axis=0)
        logger.info("Chronos forecast used.")
        return _clip_non_negative(values, point, lower, upper, "Chronos")
    except Exception as exc:  # noqa: BLE001
        logger.debug("Chronos unavailable: %s", exc)
        return None


# --- TimesFM -----------------------------------------------------------------
def _get_timesfm_model() -> Any | None:
    global _TIMESFM_MODEL
    if _TIMESFM_MODEL is not None:
        return _TIMESFM_MODEL
    try:
        import timesfm

        settings = get_settings()
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(settings.timesfm_model_id)
        model.compile(
            timesfm.ForecastConfig(
                max_context=settings.timesfm_max_context,
                max_horizon=256,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
        )
        _TIMESFM_MODEL = model
        logger.info("TimesFM model loaded (%s).", settings.timesfm_model_id)
        return model
    except Exception as exc:  # noqa: BLE001
        logger.debug("TimesFM unavailable: %s", exc)
        return None


def _try_timesfm(values: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    model = _get_timesfm_model()
    if model is None:
        return None
    try:
        inputs = [np.asarray(values, dtype=np.float32)]
        point_raw, quantiles_raw = model.forecast(horizon=horizon, inputs=inputs)
        point = np.asarray(point_raw[0] if np.ndim(point_raw) > 1 else point_raw, dtype=float).reshape(-1)
        q = np.asarray(quantiles_raw[0] if np.ndim(quantiles_raw) > 2 else quantiles_raw, dtype=float)
        if q.ndim == 2 and q.shape[0] >= 2:
            lower = np.quantile(q, 0.05, axis=0)
            upper = np.quantile(q, 0.95, axis=0)
        elif q.ndim == 1 and len(q) == len(point):
            sigma = float(np.std(values)) * 0.15 or 1.0
            lower = point - 1.645 * sigma
            upper = point + 1.645 * sigma
        else:
            sigma = float(np.std(values)) * 0.15 or 1.0
            lower = point - 1.645 * sigma
            upper = point + 1.645 * sigma
        if len(point) != horizon:
            point = np.resize(point, horizon)
            lower = np.resize(lower, horizon)
            upper = np.resize(upper, horizon)
        return point, lower, upper
    except Exception as exc:  # noqa: BLE001
        logger.warning("TimesFM forecast failed: %s", exc)
        return None


# --- Prophet -----------------------------------------------------------------
def _try_prophet(
    values: np.ndarray, dates: list[str], horizon: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    try:
        import pandas as pd
        from prophet import Prophet

        df = pd.DataFrame({"ds": pd.to_datetime(dates), "y": values})
        n = len(values)
        m = Prophet(
            daily_seasonality=n >= 14,
            weekly_seasonality=n >= 14,
            yearly_seasonality=n >= 365,
        )
        m.fit(df)
        future = m.make_future_dataframe(periods=horizon)
        fc = m.predict(future)
        point = fc["yhat"].tail(horizon).to_numpy(dtype=float)
        lower = fc["yhat_lower"].tail(horizon).to_numpy(dtype=float)
        upper = fc["yhat_upper"].tail(horizon).to_numpy(dtype=float)
        return point, lower, upper
    except Exception as exc:  # noqa: BLE001
        logger.debug("Prophet unavailable: %s", exc)
        return None


# --- ensemble / single-engine path -------------------------------------------
def _ensemble_or_single_forecast(
    values: np.ndarray, dates: list[str], horizon: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    tfm = _try_timesfm(values, horizon)
    prop = _try_prophet(values, dates, horizon)

    if tfm is not None and prop is not None:
        tf_point, tf_lower, tf_upper = tfm
        prop_point, prop_lower, prop_upper = prop
        point = (tf_point + prop_point) / 2.0
        lower = np.minimum(tf_lower, prop_lower)
        upper = np.maximum(tf_upper, prop_upper)
        return _clip_non_negative(values, point, lower, upper, "TimesFM+Prophet-ICF")
    if tfm is not None:
        point, lower, upper = tfm
        return _clip_non_negative(values, point, lower, upper, "TimesFM-ICF")
    if prop is not None:
        point, lower, upper = prop
        return _clip_non_negative(values, point, lower, upper, "Prophet")
    return _analytic_forecast(values, horizon)


def _clip_non_negative(
    values: np.ndarray,
    point: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    model_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    if values.min() >= 0:
        point = np.clip(point, 0, None)
        lower = np.clip(lower, 0, None)
    return point, lower, upper, model_name


# --- analytic fallback -------------------------------------------------------
def _analytic_forecast(values: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    n = len(values)
    x = np.arange(n)
    slope, intercept = np.polyfit(x, values, 1)
    damp = 0.97
    point = np.array([intercept + slope * (n + h) * (damp ** h) for h in range(horizon)])
    sigma = float(np.std(values - (intercept + slope * x))) or 1.0
    scale = np.sqrt(np.arange(1, horizon + 1)) * sigma * 1.645
    lower, upper = point - scale, point + scale
    return _clip_non_negative(values, point, lower, upper, "ETS-Damped-Fallback")


# --- MASE --------------------------------------------------------------------
def _mase(values: np.ndarray, dates: list[str], model_name: str) -> float:
    """Hold-out MASE: model MAE over a validation tail vs naive-1 in-sample MAE."""
    h_val = max(5, min(20, len(values) // 5))
    if len(values) <= h_val + 5:
        return 1.0
    train, test = values[:-h_val], values[-h_val:]
    train_dates = dates[:-h_val]
    pred = _holdout_predict(train, train_dates, h_val, model_name)
    mae = float(np.mean(np.abs(pred - test)))
    naive = float(np.mean(np.abs(np.diff(train)))) or 1.0
    return mae / naive


def _holdout_predict(
    train: np.ndarray, train_dates: list[str], h_val: int, model_name: str
) -> np.ndarray:
    pred: np.ndarray | None = None
    if model_name == "Chronos":
        chronos = _try_chronos(train, h_val)
        if chronos is not None:
            pred = chronos[0]
    elif model_name in {"TimesFM+Prophet-ICF", "TimesFM-ICF", "Prophet"}:
        tfm = _try_timesfm(train, h_val) if model_name in {"TimesFM+Prophet-ICF", "TimesFM-ICF"} else None
        prop = _try_prophet(train, train_dates, h_val) if model_name in {"TimesFM+Prophet-ICF", "Prophet"} else None
        if tfm is not None and prop is not None:
            pred = (tfm[0] + prop[0]) / 2.0
        elif tfm is not None:
            pred = tfm[0]
        elif prop is not None:
            pred = prop[0]
    if pred is None:
        slope, intercept = np.polyfit(np.arange(len(train)), train, 1)
        pred = intercept + slope * np.arange(len(train), len(train) + h_val)
    pred = np.asarray(pred, dtype=float).reshape(-1)
    if len(pred) != h_val:
        pred = pred[:h_val] if len(pred) > h_val else np.resize(pred, h_val)
    return pred


# --- date helpers ------------------------------------------------------------
def _future_dates(last: str, horizon: int) -> list[str]:
    try:
        start = dt.date.fromisoformat(last)
    except (ValueError, TypeError):
        start = dt.date.today()
    return [(start + dt.timedelta(days=h + 1)).strftime("%Y-%m-%d") for h in range(horizon)]


def _synth_dates(n: int) -> list[str]:
    today = dt.date.today()
    return [(today - dt.timedelta(days=(n - i))).strftime("%Y-%m-%d") for i in range(n)]
