from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

from agents.council import run, to_ready_payload
from state import CouncilOpinionPayload
from tests.fixtures import make_state


def test_council_four_opinions():
    state = make_state()
    state.metrics["deliberation"] = {
        "agreement_rate": 0.6,
        "polarization_index": 0.2,
        "narrative_clusters": [{"label": "Skeptics", "size": 10, "sentiment": -0.3}],
    }
    emitted: list[CouncilOpinionPayload] = []

    async def emit(op: CouncilOpinionPayload) -> None:
        emitted.append(op)

    mock_json = AsyncMock(
        return_value={
            "recommendation": "Proceed with phased launch.",
            "risks": ["Trust gap among skeptics"],
            "confidence": 0.72,
        }
    )

    async def _run():
        with patch("agents.council.run.get_ollama") as mock_ollama:
            mock_ollama.return_value.generate_json = mock_json
            with patch("agents.council._polish_synthesis", new=AsyncMock(return_value="Unified synthesis.")):
                return await run(state, emit)

    result = asyncio.run(_run())

    assert len(result.opinions) == 4
    assert len(emitted) == 4
    ready = to_ready_payload(result)
    assert len(ready.opinions) == 4
    assert ready.synthesis == "Unified synthesis."
    assert all(0 <= o.confidence <= 1 for o in result.opinions)


def test_council_ready_shape():
    state = make_state()
    mock_json = AsyncMock(
        return_value={"recommendation": "Test rec", "risks": [], "confidence": 0.5}
    )

    async def _run():
        with patch("agents.council.run.get_ollama") as mock_ollama:
            mock_ollama.return_value.generate_json = mock_json
            with patch("agents.council._polish_synthesis", new=AsyncMock(return_value="")):
                return await run(state, None)

    result = asyncio.run(_run())
    ready = to_ready_payload(result)
    payload = ready.model_dump()
    assert "opinions" in payload
    assert "synthesis" in payload
    for op in payload["opinions"]:
        assert op["specialist_id"] in ("pr", "brand", "marketing", "consumer_psychology")
        assert "recommendation" in op
        assert "confidence" in op
