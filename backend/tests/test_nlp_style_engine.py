"""Tests for style engine voice selection."""
from __future__ import annotations

import random

from nlp.style_engine import select_voice, topic_keyword, voice_register
from state import FacetScore, OceanScores


def test_voice_register_from_ocean():
    ocean = OceanScores(O=70, C=40, E=50, A=45, N=65)
    reg = voice_register(ocean, 42)
    assert reg in ("analyst", "skeptic", "hype", "pragmatist", "empath", "blunt", "casual")


def test_select_voice_entropy_can_shift():
    ocean = OceanScores(O=50, C=50, E=50, A=50, N=50)
    rng = random.Random(99)
    v = select_voice(ocean, 1, entropy=0.9, rng=rng)
    assert v.register


def test_topic_keyword_extracts_words():
    kw = topic_keyword("Analyze market sentiment for product launch next quarter")
    assert kw and kw != "this"


def test_persona_comment_smoke():
    from nlp.style_engine import persona_comment

    ocean = OceanScores(O=70, C=40, E=50, A=45, N=65)
    facets = [
        FacetScore(name="Anxiety", score=78.0, band="high"),
        FacetScore(name="Trust", score=25.0, band="low"),
    ]
    comment = persona_comment(
        0,
        ocean,
        facets,
        sentiment=0.2,
        cluster_label="Skeptics",
        topic="genai shopping",
    )
    assert comment and len(comment) > 10
