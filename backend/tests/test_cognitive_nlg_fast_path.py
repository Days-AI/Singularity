"""Tests for balanced-mode cognitive NLG fast path."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from agents.cognitive.response_renderer import render_population_comments
from agents.cognitive.types import (
    AgentCognitiveOutput,
    AgentMemory,
    CognitiveStateVector,
    DeliberationTrace,
    EmotionalState,
)
from config import clear_settings_cache
from state import OceanScores


def _minimal_output(agent_id: str = "p_0001") -> AgentCognitiveOutput:
    ocean = OceanScores(O=55, C=50, E=48, A=52, N=60)
    state = CognitiveStateVector(
        agent_id=agent_id,
        ocean=ocean,
        facets={"Anxiety": 70.0},
        memory=AgentMemory(),
        social_position=0.4,
        entropy_seed=1,
        cluster=0,
        cluster_label="Skeptics",
        archetype_id="arch_00",
    )
    state.emotions = EmotionalState(trust=0.5, fear=0.6, curiosity=0.4, excitement=0.3)
    state.deliberation = DeliberationTrace()
    state.confidence = 0.7
    state.total_entropy = 0.5
    return AgentCognitiveOutput(
        state=state,
        sentiment=-0.1,
        behavioral_intent="wait and see",
        key_concerns=["cost"],
        action_likelihood=0.45,
    )


@pytest.mark.asyncio
async def test_balanced_skips_polish_and_emotion(monkeypatch):
    monkeypatch.setenv("SINGULARITY_LATENCY_MODE", "balanced")
    clear_settings_cache()

    out = _minimal_output()
    polish_mock = AsyncMock()
    emotion_mock = AsyncMock(return_value=("fear", 0.9))

    with (
        patch("agents.cognitive.response_renderer.get_ollama") as go,
        patch("agents.cognitive.response_renderer.polish_persona_comment", polish_mock),
        patch("agents.cognitive.response_renderer.detect_emotion", emotion_mock),
    ):
        client = go.return_value
        client.generate_json = AsyncMock(
            return_value={"comment": "I am skeptical about this launch for now."}
        )

        await render_population_comments(
            [out],
            {0},
            "How do you feel about: genai shopping",
            "context",
            "genai shopping",
        )

    assert out.comment
    assert out.response_source == "llm"
    polish_mock.assert_not_called()
    emotion_mock.assert_not_called()
    assert out.detected_emotion == "fear"
