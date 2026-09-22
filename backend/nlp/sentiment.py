"""VADER sentiment for generated persona comments."""
from __future__ import annotations

import logging

logger = logging.getLogger("singularity.nlp.sentiment")

_analyzer = None
_analyzer_failed = False


def _get_analyzer():
    global _analyzer, _analyzer_failed
    if _analyzer_failed:
        return None
    if _analyzer is not None:
        return _analyzer
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        _analyzer = SentimentIntensityAnalyzer()
    except Exception as exc:  # noqa: BLE001
        logger.warning("VADER unavailable: %s", exc)
        _analyzer_failed = True
        _analyzer = None
    return _analyzer


def vader_available() -> bool:
    return _get_analyzer() is not None


def compound_sentiment(text: str) -> float | None:
    """Return VADER compound score in [-1, 1] or None if unavailable."""
    analyzer = _get_analyzer()
    if not analyzer or not text.strip():
        return None
    return float(analyzer.polarity_scores(text)["compound"])


def sentiment_aligned(comment: str, target: float, max_delta: float = 0.4) -> bool:
    """True when VADER compound is within max_delta of deliberation sentiment."""
    compound = compound_sentiment(comment)
    if compound is None:
        return True
    return abs(compound - target) <= max_delta
