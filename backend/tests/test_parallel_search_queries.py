"""Tests for Parallel search query derivation."""
from __future__ import annotations

from tools.search_providers import _parallel_search_queries


def test_parallel_search_queries_from_long_objective():
    objective = (
        "Do people use GenAI integration in the shopping experience? "
        "Identify key consumer pain points in the traditional online shopping journey."
    )
    queries = _parallel_search_queries(objective)
    assert 1 <= len(queries) <= 3
    assert all(len(q.split()) >= 3 for q in queries)
    assert queries[0]


def test_parallel_search_queries_short_objective():
    queries = _parallel_search_queries("genai shopping adoption rates")
    assert len(queries) == 1
    assert len(queries[0].split()) >= 3
