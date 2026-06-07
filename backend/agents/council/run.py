"""Single specialist Gemma call."""
from __future__ import annotations

import logging

from agents.council.prompts import COUNCIL_JSON_SCHEMA, SPECIALIST_PROMPTS
from config import get_settings
from llm.ollama_client import get_ollama
from state import CouncilOpinionPayload, SpecialistId

logger = logging.getLogger("singularity.council.run")


async def run_specialist(
    specialist_id: SpecialistId,
    brief: str,
    query: str,
) -> CouncilOpinionPayload:
    meta = SPECIALIST_PROMPTS[specialist_id]
    system = (
        f"You are the {meta['role']} on a Specialist Agent Council for a synthetic "
        f"market intelligence simulation. Focus: {meta['focus']}.\n"
        f"Ground recommendations in the brief. Be specific, not generic.\n{COUNCIL_JSON_SCHEMA}"
    )
    user = f"CAMPAIGN QUERY: {query}\n\nBRIEF:\n{brief}\n\nReturn your JSON now."

    recommendation, risks, confidence = "", [], 0.5
    try:
        data = await get_ollama().generate_json(
            system,
            user,
            temperature=0.45,
            max_tokens=get_settings().cognitive_deliberation_max_tokens,
        )
        recommendation = str(data.get("recommendation", ""))[:500]
        raw_risks = data.get("risks", [])
        risks = [str(r)[:120] for r in raw_risks][:4] if isinstance(raw_risks, list) else []
        confidence = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Council specialist %s fallback: %s", specialist_id, exc)
        recommendation = f"Proceed cautiously on '{query[:80]}' from a {meta['role']} lens."
        risks = ["Insufficient model output; using heuristic fallback"]

    return CouncilOpinionPayload(
        specialist_id=specialist_id,
        role=meta["role"],
        recommendation=recommendation,
        risks=risks,
        confidence=round(confidence, 3),
    )
