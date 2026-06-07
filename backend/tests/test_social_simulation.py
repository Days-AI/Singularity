from __future__ import annotations

import asyncio

import pytest

from agents.social.simulation import run
from state import SocialInteractionTickPayload
from tests.fixtures import make_state


def test_social_simulation_multi_round(monkeypatch):
    monkeypatch.setenv("SOCIAL_SIMULATION_ROUNDS", "3")
    from config import get_settings

    get_settings.cache_clear()

    state = make_state()
    ticks: list[SocialInteractionTickPayload] = []

    async def emit_tick(payload: SocialInteractionTickPayload) -> None:
        ticks.append(payload)

    result = asyncio.run(run(state, emit_tick))

    assert len(ticks) == 3
    assert result.metrics["rounds_completed"] == 3
    assert result.final_payload is not None
    assert result.final_payload.rounds_completed == 3
    assert "deliberation_refresh" in result.metrics
    assert ticks[0].round == 1
    assert ticks[-1].round == 3


def test_social_polarization_drift(monkeypatch):
    monkeypatch.setenv("SOCIAL_SIMULATION_ROUNDS", "2")
    from config import get_settings

    get_settings.cache_clear()

    state = make_state()
    result = asyncio.run(run(state, None))
    pol_start = result.rounds[0].polarization_index if result.rounds else 0.0
    pol_end = result.metrics.get("polarization_index", 0.0)
    assert isinstance(pol_start, float)
    assert isinstance(pol_end, float)


def test_social_empty_opinions():
    from state import SingularityState

    state = SingularityState(query="test", flow_uuid="empty")
    result = asyncio.run(run(state, None))
    assert result.metrics == {}
    assert result.final_payload is None
