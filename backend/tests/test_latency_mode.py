"""Tests for SINGULARITY_LATENCY_MODE presets."""
from __future__ import annotations

import os

import pytest

from config import Settings, clear_settings_cache, get_settings


@pytest.fixture(autouse=True)
def _reset_settings_cache():
    clear_settings_cache()
    yield
    clear_settings_cache()


def test_balanced_preset_sample_size(monkeypatch):
    monkeypatch.setenv("SINGULARITY_LATENCY_MODE", "balanced")
    monkeypatch.setenv("COGNITIVE_LLM_SAMPLE_SIZE", "750")
    clear_settings_cache()
    s = get_settings()
    assert s.latency_mode_normalized == "balanced"
    assert s.cognitive_llm_sample_size == 150
    assert s.persona_polish_openrouter is False
    assert s.decision_engine_llm_enrich is False
    assert s.skip_cognitive_nlg_extras is True
    assert s.flow_budget_seconds == 600
    assert s.naturalize_pack_size == 6
    assert s.programmatic_nlg_max_tokens == 72


def test_fast_preset_sample_size(monkeypatch):
    monkeypatch.setenv("SINGULARITY_LATENCY_MODE", "fast")
    clear_settings_cache()
    s = get_settings()
    assert s.cognitive_llm_sample_size == 96
    assert s.monte_carlo_simulations == 800


def test_quality_mode_keeps_env_sample_size(monkeypatch):
    monkeypatch.setenv("SINGULARITY_LATENCY_MODE", "quality")
    monkeypatch.setenv("COGNITIVE_LLM_SAMPLE_SIZE", "500")
    monkeypatch.setenv("PERSONA_POLISH_OPENROUTER", "true")
    clear_settings_cache()
    s = get_settings()
    assert s.cognitive_llm_sample_size == 500
    assert s.persona_polish_openrouter is True
    assert s.skip_cognitive_nlg_extras is False


def test_cognitive_concurrency_clamped_to_ollama(monkeypatch):
    monkeypatch.setenv("SINGULARITY_LATENCY_MODE", "quality")
    monkeypatch.setenv("OLLAMA_CONCURRENCY", "4")
    monkeypatch.setenv("COGNITIVE_LLM_CONCURRENCY", "12")
    clear_settings_cache()
    s = get_settings()
    assert s.cognitive_llm_concurrency == 4


def test_flow_budget_exceeded(monkeypatch):
    monkeypatch.setenv("SINGULARITY_LATENCY_MODE", "quality")
    clear_settings_cache()
    s = Settings(flow_budget_seconds=600)
    assert s.flow_budget_exceeded(599_999) is False
    assert s.flow_budget_exceeded(600_000) is True
