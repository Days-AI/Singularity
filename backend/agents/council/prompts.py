"""Council specialist prompts."""
from __future__ import annotations

SPECIALIST_PROMPTS: dict[str, dict[str, str]] = {
    "pr": {
        "role": "PR Strategist",
        "focus": "reputation, media narrative, crisis framing, stakeholder trust",
    },
    "brand": {
        "role": "Brand Director",
        "focus": "brand equity, positioning, trust, long-term identity fit",
    },
    "marketing": {
        "role": "Marketing Lead",
        "focus": "campaign mechanics, channels, conversion, audience targeting",
    },
    "consumer_psychology": {
        "role": "Consumer Psychologist",
        "focus": "OCEAN/facet behavioral barriers, adoption psychology, cognitive biases",
    },
}

COUNCIL_JSON_SCHEMA = """Respond ONLY with valid JSON:
{
  "recommendation": "<2-3 sentences, actionable>",
  "risks": ["<risk1>", "<risk2>"],
  "confidence": <float 0.0..1.0>
}"""
