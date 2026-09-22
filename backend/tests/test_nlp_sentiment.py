"""Tests for VADER sentiment reconciliation."""
from __future__ import annotations

from nlp.sentiment import compound_sentiment, sentiment_aligned


def test_vader_compound_positive():
    score = compound_sentiment("I love this product launch, it is amazing!")
    if score is None:
        return  # vader not installed in CI
    assert score > 0.3


def test_sentiment_aligned_within_delta():
    comment = "I really like this idea and feel positive about it."
    assert sentiment_aligned(comment, 0.5, max_delta=0.6)
