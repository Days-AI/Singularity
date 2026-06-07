from __future__ import annotations

import re

from agents.psychometric import (
    _FACET_REACTIONS,
    _REGISTER_PREFIX,
    _persona_comment,
)
from state import FacetScore, OceanScores

_BANNED_PATTERNS = [
    r"\blowkey\b",
    r"\bngl\b",
    r"\btbh\b",
    r"\breal talk\b",
    r"\bcolor me\b",
    r"\bnot gonna lie\b",
    r"\bstraight up\b",
    r"\byeah so\b",
    r"#\w",
]


def _contains_banned(text: str) -> str | None:
    lower = text.lower()
    for pattern in _BANNED_PATTERNS:
        if re.search(pattern, lower):
            return pattern
    return None


def test_register_prefixes_avoid_slang():
    for register, prefixes in _REGISTER_PREFIX.items():
        for prefix in prefixes:
            hit = _contains_banned(prefix)
            assert hit is None, f"register {register!r} prefix {prefix!r} matches {hit}"


def test_facet_reactions_avoid_slang():
    for facet, bands in _FACET_REACTIONS.items():
        for band, lines in bands.items():
            for line in lines:
                hit = _contains_banned(line)
                assert hit is None, f"facet {facet}/{band} line {line!r} matches {hit}"


def test_persona_comment_smoke_no_slang():
    sample_oceans = [
        OceanScores(O=75, C=70, E=65, A=45, N=62),
        OceanScores(O=40, C=55, E=72, N=35, A=68),
        OceanScores(O=50, C=50, E=50, A=50, N=50),
    ]
    sample_facets = [
        [
            FacetScore(name="Anxiety", score=78.0, band="high"),
            FacetScore(name="Trust", score=25.0, band="low"),
        ],
        [
            FacetScore(name="Excitement", score=80.0, band="high"),
            FacetScore(name="Cautiousness", score=30.0, band="low"),
        ],
        [
            FacetScore(name="Intellect", score=52.0, band="moderate"),
            FacetScore(name="Orderliness", score=48.0, band="moderate"),
        ],
    ]
    for p_idx in range(50):
        ocean = sample_oceans[p_idx % len(sample_oceans)]
        facets = sample_facets[p_idx % len(sample_facets)]
        comment = _persona_comment(
            p_idx,
            ocean,
            facets,
            sentiment=0.2 if p_idx % 3 == 0 else -0.1 if p_idx % 3 == 1 else 0.0,
            cluster_label="Pragmatists",
            topic="value perception",
            arch_intent="evaluating options carefully",
            arch_emotion="measured",
        )
        assert comment
        hit = _contains_banned(comment)
        assert hit is None, f"persona {p_idx} comment {comment!r} matches {hit}"
        assert "#" not in comment
