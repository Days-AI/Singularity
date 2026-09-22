"""Tests for NLP diversity sampling params."""
from __future__ import annotations

from nlp.diversity import sampling_params


def test_sampling_params_entropy_increases_temperature():
    low = sampling_params(0.1, 30.0, "analyst")
    high = sampling_params(0.9, 70.0, "hype")
    assert high.temperature > low.temperature
    assert high.top_p >= low.top_p


def test_sampling_params_bounded():
    p = sampling_params(1.5, 100.0, "blunt")
    assert 0.2 <= p.temperature <= 1.2
    assert 0.75 <= p.top_p <= 0.99
