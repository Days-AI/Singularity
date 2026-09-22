"""Persona polish fallback tests."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

from nlp.polish import polish_persona_comment


def test_polish_returns_original_when_disabled():
    comment = "I'm cautiously optimistic about this launch."
    with patch("nlp.polish.get_settings") as gs:
        gs.return_value.persona_polish_openrouter = False
        gs.return_value.openrouter_api_key = None
        out, src = asyncio.run(
            polish_persona_comment(
                comment,
                ocean_summary="O50 C50 E50 A50 N50",
                topic="product launch",
                voice_register="casual",
                sentiment=0.4,
                intent="evaluate",
            )
        )
    assert out == comment
    assert src == "llm"
