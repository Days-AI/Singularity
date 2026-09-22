"""Tests for draft-based NLG pipeline (no phrase banks)."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from agents.cognitive.types import AgentCognitiveOutput, CognitiveStateVector
from nlp.style_engine import persona_comment
from state import FacetScore, OceanScores


def test_style_engine_has_no_phrase_banks():
    from nlp import style_engine as se

    assert not hasattr(se, "_INTENT_REALIZATIONS")
    assert not hasattr(se, "_FACET_REACTIONS")


def test_persona_comment_compositional_fallback():
    ocean = OceanScores(O=50, C=55, E=45, A=50, N=72)
    facets = [FacetScore(name="Anxiety", score=80.0, band="high")]
    comment = persona_comment(
        7, ocean, facets, sentiment=-0.2, cluster_label="Skeptics",
        topic="product launch", entropy=0.55,
    )
    assert comment
    assert "practically speaking" not in comment.lower()


@pytest.mark.asyncio
async def test_compose_programmatic_calls_naturalize():
    from nlp.style_engine import compose_programmatic

    ocean = OceanScores(O=70, C=65, E=45, A=50, N=40)
    state = CognitiveStateVector(
        agent_id="p_0042",
        ocean=ocean,
        facets={"Intellect": 80.0, "Cautiousness": 72.0},
        cluster_label="Analysts",
        entropy_seed=42,
        cluster=1,
        archetype_id="arch_01",
        total_entropy=0.6,
    )
    out = AgentCognitiveOutput(
        state=state,
        sentiment=0.1,
        behavioral_intent="evaluating evidence carefully",
        key_concerns=["proof"],
        action_likelihood=0.5,
    )
    with patch("nlp.naturalize.naturalize_draft", new_callable=AsyncMock) as nm:
        nm.return_value = ("I'd want to see the reasoning first.", "naturalized")
        import random
        text, src = await compose_programmatic(out, None, random.Random(99), stimulus="test query")
    assert text
    assert src == "naturalized"
    nm.assert_called_once()
