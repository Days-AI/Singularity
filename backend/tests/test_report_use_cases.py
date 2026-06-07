from __future__ import annotations

from report.evidence_brief import build_web_intelligence
from report.generate import _structured_data, _to_sections
from report.use_cases import (
    USE_CASES,
    build_application_playbooks_markdown,
    merge_llm_playbooks,
)
from state import EvidenceItem, SingularityState


def test_use_cases_catalog_has_six_domains():
    assert len(USE_CASES) == 6
    domains = {uc.domain for uc in USE_CASES}
    assert "Communications & PR" in domains
    assert "Public Policy" in domains


def test_build_application_playbooks_includes_canonical_descriptions():
    data = {
        "query": "Test launch narrative",
        "population": 1500,
        "mean_sentiment": 0.12,
    }
    md = build_application_playbooks_markdown(data)
    for uc in USE_CASES:
        assert uc.domain in md
        assert uc.tagline in md
        assert uc.description in md
        assert "**Simulation insight:**" in md


def test_build_application_playbooks_merges_llm_recommendations():
    data = {"query": "q", "population": 1500, "mean_sentiment": 0.0}
    md = build_application_playbooks_markdown(
        data,
        {"Product": "Ship a limited beta to Enthusiasts first."},
    )
    assert "**Recommended action:** Ship a limited beta to Enthusiasts first." in md


def test_merge_llm_playbooks_extracts_domains():
    polished = {
        "application_playbooks": [
            {"domain": "Marketing", "recommendation": "Run A/B on proof-point creative."},
            {"domain": "Branding", "recommendation": "Test warm vs bold voice."},
        ]
    }
    merged = merge_llm_playbooks(polished)
    assert merged["Marketing"] == "Run A/B on proof-point creative."
    assert merged["Branding"] == "Test warm vs bold voice."


def test_to_sections_emits_seven_sections():
    state = SingularityState(flow_uuid="test-flow", query="q")
    state.evidence = [
        EvidenceItem(
            source="Serper",
            title="Market update",
            detail="Growth continues.",
            sentiment=0.3,
        ),
    ]
    data = _structured_data(state)
    polished = {
        "executive_summary": "Summary.",
        "external_intelligence_summary": "One cited source supports growth.",
        "key_findings": ["Finding one."],
        "strategic_implications": "Implications.",
        "risk_flags": ["Risk one."],
    }
    sections = _to_sections(polished, data)
    assert len(sections) == 7
    assert sections[1].section == "External Intelligence & Sources"
    assert sections[5].section == "Simulation Applications"
    assert sections[6].section == "Council Consensus"
    assert sections[0].total == 7
    assert sections[6].index == 6
    assert "Communications & PR" in sections[5].content
    assert len(build_web_intelligence(state)["ranked_findings"]) >= 1
    assert "Serper" in sections[1].content
