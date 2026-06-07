"""Tests for report evidence brief builder."""
from __future__ import annotations

from report.evidence_brief import (
    build_external_intelligence_md,
    build_web_intelligence,
    findings_to_highlights,
)
from report.generate import _structured_data, _to_sections
from state import EvidenceItem, OceanScores, PersonaResponse, SingularityState


def _state_with_evidence() -> SingularityState:
    state = SingularityState(flow_uuid="test-flow", query="EV adoption in Europe")
    state.evidence = [
        EvidenceItem(
            source="Serper",
            title="EV sales surge in Q1",
            detail="Battery EV registrations rose 18% year-on-year.",
            url="https://example.com/ev",
            sentiment=0.4,
        ),
        EvidenceItem(
            source="DuckDuckGo",
            title="Charging infrastructure gaps",
            detail="Rural coverage remains a adoption barrier.",
            sentiment=-0.25,
        ),
        EvidenceItem(
            source="Simulation",
            title="Population sentiment",
            detail="Mean sentiment +0.12 across personas.",
            sentiment=0.12,
        ),
    ]
    state.persona_responses = [
        PersonaResponse(
            archetype_id="a1",
            ocean=OceanScores(O=50, C=50, E=50, A=50, N=50),
            sentiment_score=0.15,
        ),
        PersonaResponse(
            archetype_id="a2",
            ocean=OceanScores(O=55, C=45, E=48, A=52, N=40),
            sentiment_score=0.09,
        ),
    ]
    return state


def test_build_web_intelligence_source_breakdown():
    intel = build_web_intelligence(_state_with_evidence(), max_items=12)
    assert intel["external_count"] == 2
    assert intel["internal_count"] == 1
    assert len(intel["source_breakdown"]) >= 2
    serper = next(b for b in intel["source_breakdown"] if b["source"] == "Serper")
    assert serper["count"] == 1
    assert serper["mean_sentiment"] == 0.4


def test_ranked_findings_include_url_and_sentiment():
    intel = build_web_intelligence(_state_with_evidence())
    ranked = intel["ranked_findings"]
    assert len(ranked) >= 2
    with_url = next(f for f in ranked if f.get("url"))
    assert with_url["url"] == "https://example.com/ev"
    assert "sentiment" in with_url


def test_simulation_alignment():
    intel = build_web_intelligence(_state_with_evidence())
    align = intel["simulation_alignment"]
    assert align is not None
    assert align["label"] in ("aligned", "divergent")
    assert "delta" in align


def test_build_external_intelligence_md_structure():
    state = _state_with_evidence()
    intel = build_web_intelligence(state)
    md = build_external_intelligence_md(intel, state.query, llm_summary="Synthesis paragraph.")
    assert "**Situation.**" in md
    assert "**Key external signals**" in md
    assert "Synthesis paragraph." in md
    assert "Serper" in md
    assert "https://example.com/ev" in md


def test_findings_to_highlights_compat():
    intel = build_web_intelligence(_state_with_evidence())
    highlights = findings_to_highlights(intel["ranked_findings"])
    assert highlights[0]["source"]
    assert "title" in highlights[0]


def test_structured_data_includes_web_intelligence():
    data = _structured_data(_state_with_evidence())
    assert "web_intelligence" in data
    assert len(data["evidence_highlights"]) >= 2


def test_to_sections_emits_seven_sections_with_external_intelligence():
    state = _state_with_evidence()
    data = _structured_data(state)
    polished = {
        "executive_summary": "Summary.",
        "external_intelligence_summary": "External signals support cautious optimism.",
        "key_findings": ["**Insight** — EV growth (Source: Serper) — expand reach."],
        "strategic_implications": "Implications.",
        "risk_flags": ["Risk one."],
    }
    sections = _to_sections(polished, data)
    assert len(sections) == 7
    assert sections[0].total == 7
    assert sections[1].section == "External Intelligence & Sources"
    assert "Serper" in sections[1].content
    assert sections[5].section == "Simulation Applications"


def test_empty_evidence_coverage_note():
    state = SingularityState(flow_uuid="test-flow", query="test")
    intel = build_web_intelligence(state)
    assert "No live web evidence" in intel["coverage_note"]
    md = build_external_intelligence_md(intel, "test")
    assert "No external intelligence" in md or "No live web" in md
