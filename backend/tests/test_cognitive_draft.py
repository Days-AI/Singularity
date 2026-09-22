"""Tests for cognitive draft builder."""
from __future__ import annotations

from agents.cognitive.types import AgentCognitiveOutput, CognitiveStateVector, DeliberationTrace
from nlp.cognitive_draft import build_cognitive_draft, draft_fallback_line
from state import OceanScores


def _output() -> AgentCognitiveOutput:
    ocean = OceanScores(O=70, C=65, E=45, A=50, N=40)
    state = CognitiveStateVector(
        agent_id="p_0007",
        ocean=ocean,
        facets={"Intellect": 80.0, "Cautiousness": 72.0},
        cluster_label="Analysts",
        entropy_seed=7,
        cluster=1,
        archetype_id="arch_01",
        confidence=0.72,
        total_entropy=0.55,
    )
    state.deliberation = DeliberationTrace(
        winning_voice="Risk Aversion",
        utterances=[],
    )
    return AgentCognitiveOutput(
        state=state,
        sentiment=-0.12,
        behavioral_intent="waiting for better value signals",
        key_concerns=["cost and value", "downside risk"],
        action_likelihood=0.42,
    )


def test_build_cognitive_draft_from_output():
    out = _output()
    draft = build_cognitive_draft(out, None, "How do consumers view GenAI shopping?", voice_register="analyst")
    assert draft.agent_id == "p_0007"
    assert draft.winning_voice == "Risk Aversion"
    assert draft.behavioral_intent
    assert "cost" in " ".join(draft.key_concerns).lower()
    assert draft.voice_register == "analyst"
    assert draft.sentiment == -0.12


def test_draft_fallback_line_no_phrase_bank():
    out = _output()
    draft = build_cognitive_draft(out, None, "GenAI shopping assistants", voice_register="skeptic")
    line = draft_fallback_line(draft)
    assert line
    assert "practically speaking" not in line.lower()
    assert "on genai" not in line.lower()
