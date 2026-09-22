"""Tests for topic context fallbacks."""
from __future__ import annotations

import asyncio

from nlp.topic_context import TopicContext, build_topic_context
from state import EvidenceItem


def test_topic_context_fallback_keyphrases():
    ctx = TopicContext(query="How will consumers react to electric vehicle pricing?")
    assert ctx.query


def test_build_topic_context_disabled():
    from config import get_settings

    settings = get_settings()
    if settings.nlp_pipeline_enabled:
        return
    ctx = asyncio.get_event_loop().run_until_complete(
        build_topic_context("test query", [])
    )
    assert ctx.domain_label == "general"


def test_build_topic_context_with_evidence():
    evidence = [
        EvidenceItem(source="News", title="EV pricing trends", detail="Battery costs falling"),
    ]
    ctx = asyncio.run(build_topic_context("electric vehicle market sentiment", evidence))
    assert ctx.query
    assert isinstance(ctx.keyphrases, list)
