"""Tests for Gemma naturalize layer."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from nlp.cognitive_draft import CognitiveDraft
from nlp.naturalize import naturalize_draft, naturalize_drafts_batch


def _draft() -> CognitiveDraft:
    return CognitiveDraft(
        agent_id="p_0001",
        topic="consumer trends",
        domain_label="marketing",
        stimulus_snippet="GenAI shopping",
        sentiment=-0.1,
        confidence=0.7,
        behavioral_intent="evaluating evidence",
        winning_voice="Risk Aversion",
        key_concerns=["downside risk"],
        emotions={"trust": 0.4, "fear": 0.6, "curiosity": 0.5, "excitement": 0.3},
        deliberation_summary=['Risk Aversion: "-0.20" (w=0.55)'],
        ocean_summary="O55 C60 E48 A52 N65",
        voice_register="skeptic",
        entropy=0.5,
    )


@pytest.mark.asyncio
async def test_naturalize_draft_ollama_success():
    with patch("nlp.naturalize.get_ollama") as go:
        client = go.return_value
        client.generate_json = AsyncMock(
            return_value={"comment": "Maybe I'm overthinking it, but I'd want to see the downside first."}
        )
        comment, src = await naturalize_draft(_draft())
    assert "downside" in comment.lower() or len(comment) > 10
    assert src == "naturalized"


@pytest.mark.asyncio
async def test_naturalize_drafts_batch_pack_success(monkeypatch):
    from config import clear_settings_cache

    monkeypatch.setenv("SINGULARITY_LATENCY_MODE", "quality")
    monkeypatch.setenv("NATURALIZE_PACK_SIZE", "2")
    clear_settings_cache()
    drafts = [_draft(), _draft()]
    drafts[1].agent_id = "p_0002"
    with patch("nlp.naturalize._ollama_naturalize_pack", new_callable=AsyncMock) as pack:
        pack.return_value = ["First comment here.", "Second comment here."]
        results, stats = await naturalize_drafts_batch(drafts)
    assert len(results) == 2
    assert all(src == "naturalized" for _, src in results)
    assert stats["pack_size"] == 2
    pack.assert_awaited_once()


@pytest.mark.asyncio
async def test_naturalize_draft_fallback_on_failure():
    with patch("nlp.naturalize.get_ollama") as go:
        client = go.return_value
        client.generate_json = AsyncMock(side_effect=RuntimeError("ollama down"))
        comment, src = await naturalize_draft(_draft())
    assert comment
    assert src == "draft_fallback"
