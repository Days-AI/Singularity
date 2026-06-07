from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from agents import forecast as forecast_agent
from state import TimeSeries


def _synthetic_series(n: int = 60, horizon: int = 90) -> TimeSeries:
    rng = np.random.default_rng(42)
    values = (100 + np.cumsum(rng.normal(0.5, 1.0, n))).tolist()
    dates = [f"2025-01-{i + 1:02d}" for i in range(n)]
    return TimeSeries(name="Test Index", dates=dates, values=values)


def test_forecast_blocking_shape_and_horizon():
    series = _synthetic_series()
    horizon = 90
    with patch.object(forecast_agent, "_try_chronos", return_value=None), patch.object(
        forecast_agent, "_try_timesfm", return_value=None
    ), patch.object(forecast_agent, "_try_prophet", return_value=None):
        payload = forecast_agent._forecast_blocking(series, horizon)

    assert payload.horizon_days == horizon
    assert payload.model == "ETS-Damped-Fallback"
    assert len(payload.history) <= 60
    assert len(payload.predictions) == horizon
    assert len(payload.intervals) == horizon
    assert payload.mase_score >= 0.0
    for pred, band in zip(payload.predictions, payload.intervals):
        assert pred.date == band.date
        assert band.lower <= band.upper


def test_ensemble_blends_timesfm_and_prophet():
    series = _synthetic_series(n=40, horizon=30)
    horizon = 30
    tf_point = np.arange(horizon, dtype=float) + 10.0
    tf_lower = tf_point - 2.0
    tf_upper = tf_point + 2.0
    prop_point = np.arange(horizon, dtype=float) + 20.0
    prop_lower = prop_point - 3.0
    prop_upper = prop_point + 3.0

    with patch.object(forecast_agent, "_try_chronos", return_value=None), patch.object(
        forecast_agent, "_try_timesfm", return_value=(tf_point, tf_lower, tf_upper)
    ), patch.object(
        forecast_agent, "_try_prophet", return_value=(prop_point, prop_lower, prop_upper)
    ):
        payload = forecast_agent._forecast_blocking(series, horizon)

    assert payload.model == "TimesFM+Prophet-ICF"
    expected_point = (tf_point + prop_point) / 2.0
    np.testing.assert_allclose(
        [p.value for p in payload.predictions], expected_point, rtol=1e-3
    )
    for band, lo, hi in zip(payload.intervals, np.minimum(tf_lower, prop_lower), np.maximum(tf_upper, prop_upper)):
        assert band.lower == pytest.approx(float(lo), rel=1e-3)
        assert band.upper == pytest.approx(float(hi), rel=1e-3)


def test_prophet_only_fallback_label():
    series = _synthetic_series(n=40, horizon=20)
    horizon = 20
    prop_point = np.linspace(100, 110, horizon)
    prop_lower = prop_point - 1.0
    prop_upper = prop_point + 1.0

    with patch.object(forecast_agent, "_try_chronos", return_value=None), patch.object(
        forecast_agent, "_try_timesfm", return_value=None
    ), patch.object(
        forecast_agent, "_try_prophet", return_value=(prop_point, prop_lower, prop_upper)
    ):
        payload = forecast_agent._forecast_blocking(series, horizon)

    assert payload.model == "Prophet"
    np.testing.assert_allclose([p.value for p in payload.predictions], prop_point, rtol=1e-3)


def test_mase_finite_for_ensemble():
    values = np.linspace(100, 130, 50)
    dates = [f"2025-01-{i + 1:02d}" for i in range(len(values))]
    h_val = 10

    def fake_timesfm(train: np.ndarray, horizon: int):
        return np.linspace(train[-1], train[-1] + horizon, horizon), train[-1] - 1, train[-1] + 1

    def fake_prophet(train: np.ndarray, train_dates: list[str], horizon: int):
        return np.linspace(train[-1], train[-1] + horizon * 0.9, horizon), train[-1] - 2, train[-1] + 2

    with patch.object(forecast_agent, "_try_timesfm", side_effect=fake_timesfm), patch.object(
        forecast_agent, "_try_prophet", side_effect=fake_prophet
    ):
        score = forecast_agent._mase(values, dates, "TimesFM+Prophet-ICF")

    assert np.isfinite(score)
    assert score > 0.0
    pred = forecast_agent._holdout_predict(values[:-h_val], dates[:-h_val], h_val, "TimesFM+Prophet-ICF")
    assert len(pred) == h_val
