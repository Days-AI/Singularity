"""Run one SingularityFlow end-to-end and print elapsed_ms (for latency verification)."""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from config import get_settings
from flow import SingularityFlow
from sse import EventBus
from state import SingularityState


async def main() -> None:
    settings = get_settings()
    print(
        f"mode={settings.latency_mode_normalized} "
        f"sample={settings.cognitive_llm_sample_size} "
        f"ollama_concurrency={settings.ollama_concurrency} "
        f"budget={settings.flow_budget_seconds}s"
    )
    state = SingularityState(
        query="How do consumers perceive GenAI shopping assistants?",
        flow_uuid="e2e-balanced-verify",
        web_sources_enabled=False,
    )
    bus = EventBus()
    flow = SingularityFlow(state=state, bus=bus)
    t0 = time.monotonic()
    await flow.run()
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    llm_n = state.metrics.get("llm_sample_count") or state.metrics.get("deliberation", {}).get(
        "llm_sample_count"
    )
    print(f"flow_complete elapsed_ms={elapsed_ms} under_600s={elapsed_ms < 600_000}")
    print(f"llm_sample_count={llm_n}")
    print(f"persona_opinions={len(state.persona_opinions)}")


if __name__ == "__main__":
    asyncio.run(main())
