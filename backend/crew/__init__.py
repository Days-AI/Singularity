"""CrewAI synthesis layer (Phase 2).

Produces the strategic report via a small crew of role-specialized agents that
are injected with aggregated persona OCEAN/facet context. Local Ollama/Gemma is
the generation engine; OpenRouter polishes the final editorial pass.

Import-guarded and flag-gated (`CREWAI_ENABLED`): when crewai/litellm are absent
or the crew fails, callers fall back to the existing two-stage report builder.
"""
from __future__ import annotations

from crew.personas import build_persona_context, render_persona_context

__all__ = ["build_persona_context", "render_persona_context", "synthesize", "crew_available"]


def crew_available() -> bool:
    from crew.crew import crew_available as _ca

    return _ca()


async def synthesize(state, structured_data):
    from crew.crew import synthesize as _synth

    return await _synth(state, structured_data)
