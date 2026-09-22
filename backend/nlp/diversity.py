"""Entropy-driven LLM sampling parameters."""
from __future__ import annotations

from dataclasses import dataclass

from config import get_settings


@dataclass
class SamplingParams:
    temperature: float
    top_p: float
    repeat_penalty: float


def sampling_params(
    entropy: float,
    neuroticism: float,
    voice_register: str = "casual",
) -> SamplingParams:
    """Map personality entropy + voice type to Ollama generation knobs."""
    settings = get_settings()
    e = max(0.0, min(1.0, entropy))
    n = max(0.0, min(100.0, neuroticism))

    temp = 0.45 + e * 0.45 + n / 400.0
    top_p = settings.persona_top_p_min + e * (settings.persona_top_p_max - settings.persona_top_p_min)
    repeat_penalty = 1.05 + (1.0 - e) * 0.08

    # Analyst/pragmatist voices: tighter sampling for consistency.
    if voice_register in ("analyst", "pragmatist"):
        temp -= 0.08
        top_p -= 0.03
    elif voice_register in ("hype", "blunt"):
        temp += 0.05
        top_p += 0.02

    return SamplingParams(
        temperature=round(max(0.2, min(1.2, temp)), 2),
        top_p=round(max(0.75, min(0.99, top_p)), 3),
        repeat_penalty=round(max(1.0, min(1.25, repeat_penalty)), 2),
    )
