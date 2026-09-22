"""Local Gemma naturalization of cognitive drafts into first-person comments."""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any

from config import get_settings
from llm.ollama_client import get_ollama
from llm.openrouter_client import get_openrouter
from nlp.cognitive_draft import CognitiveDraft, draft_fallback_line
from nlp.diversity import sampling_params
from nlp.prompt_budget import clamp_text, fit_prompt_pair
from prompts import PERSONA_POLISH_SYSTEM, PERSONA_POLISH_USER

logger = logging.getLogger("singularity.nlp.naturalize")

NATURALIZE_SYSTEM = """You are one synthetic citizen in a population simulation.
Turn the brief into ONE first-person comment (1-2 sentences).
Preserve sentiment and belief voice. Sound natural; contractions OK.
No persona IDs, slang, or bullet lists. Do not invent facts.
Respond ONLY with JSON: {"comment": "<first-person comment>"}"""

NATURALIZE_PACK_SYSTEM = """You are a population simulator. For each numbered citizen brief,
write ONE unique first-person comment (1-2 sentences).
Preserve each brief's sentiment and belief voice. Sound human; contractions OK.
No IDs, slang, or bullets. Do not invent facts.
Respond ONLY with JSON: {"comments": ["...", "..."]} — same count and order as briefs."""


def _naturalize_ctx() -> int:
    settings = get_settings()
    return (
        settings.naturalize_ollama_num_ctx
        or settings.cognitive_ollama_num_ctx
        or settings.ollama_num_ctx
    )


def _extract_comment(raw: str) -> str:
    block = re.search(r"\{.*\}", raw, re.DOTALL)
    if not block:
        return ""
    try:
        data = json.loads(block.group(0))
    except json.JSONDecodeError:
        return ""
    return str(data.get("comment", "")).strip()


def _extract_pack_comments(data: dict[str, Any] | str, expected: int) -> list[str]:
    if isinstance(data, str):
        block = re.search(r"\{.*\}", data, re.DOTALL)
        if not block:
            return []
        try:
            data = json.loads(block.group(0))
        except json.JSONDecodeError:
            return []
    comments = data.get("comments")
    if not isinstance(comments, list):
        return []
    out: list[str] = []
    for item in comments[:expected]:
        text = str(item).strip()
        if text and len(text) >= 10:
            out.append(text[:220])
        else:
            out.append("")
    return out


async def _ollama_naturalize(
    system: str,
    user: str,
    *,
    draft: CognitiveDraft,
    max_tokens: int,
) -> str:
    ctx = _naturalize_ctx()
    _, user_fit, _ = fit_prompt_pair(system, user, num_ctx=ctx, num_predict=max_tokens)
    n_match = re.search(r"N(\d+)", draft.ocean_summary)
    n_score = float(n_match.group(1)) if n_match else 50.0
    params = sampling_params(draft.entropy, n_score, draft.voice_register)
    data = await get_ollama().generate_json(
        system,
        user_fit,
        temperature=params.temperature,
        top_p=params.top_p,
        repeat_penalty=params.repeat_penalty,
        max_tokens=max_tokens,
        num_ctx=ctx,
    )
    comment = str(data.get("comment", "")).strip()
    if comment and len(comment) >= 10:
        return comment[:220]
    return ""


async def _ollama_naturalize_pack(
    drafts: list[CognitiveDraft],
    *,
    max_tokens: int,
) -> list[str]:
    if not drafts:
        return []
    if len(drafts) == 1:
        brief = drafts[0].to_naturalize_brief()
        user = f"CITIZEN BRIEF:\n{brief}\n\nWrite the JSON comment now."
        comment = await _ollama_naturalize(
            NATURALIZE_SYSTEM, user, draft=drafts[0], max_tokens=max_tokens
        )
        return [comment]

    lines = []
    for idx, draft in enumerate(drafts, start=1):
        lines.append(f"[{idx}]\n{draft.to_naturalize_brief()}")
    user = "CITIZEN BRIEFS:\n" + "\n\n".join(lines) + "\n\nWrite the JSON comments now."
    ctx = _naturalize_ctx()
    pack_tokens = min(
        max_tokens * len(drafts) + 32,
        get_settings().naturalize_pack_max_tokens,
    )
    _, user_fit, _ = fit_prompt_pair(
        NATURALIZE_PACK_SYSTEM, user, num_ctx=ctx, num_predict=pack_tokens
    )
    rep = drafts[0]
    n_match = re.search(r"N(\d+)", rep.ocean_summary)
    n_score = float(n_match.group(1)) if n_match else 50.0
    mean_entropy = sum(d.entropy for d in drafts) / len(drafts)
    params = sampling_params(mean_entropy, n_score, rep.voice_register)
    data = await get_ollama().generate_json(
        NATURALIZE_PACK_SYSTEM,
        user_fit,
        temperature=params.temperature,
        top_p=params.top_p,
        repeat_penalty=params.repeat_penalty,
        max_tokens=pack_tokens,
        num_ctx=ctx,
    )
    comments = _extract_pack_comments(data, len(drafts))
    if len(comments) == len(drafts) and all(comments):
        return comments
    return []


async def naturalize_draft(
    draft: CognitiveDraft,
    *,
    raw_comment: str | None = None,
) -> tuple[str, str]:
    """Return (comment, source) where source is naturalized, llm_polished, or draft_fallback."""
    settings = get_settings()
    if settings.programmatic_nlg_mode == "draft_only":
        return draft_fallback_line(draft), "draft_fallback"

    brief = draft.to_naturalize_brief()
    if raw_comment:
        user = PERSONA_POLISH_USER.format(
            comment=clamp_text(raw_comment, 180),
            ocean=draft.ocean_summary,
            topic=draft.topic,
            voice=draft.voice_register,
            sentiment=f"{draft.sentiment:+.2f}",
            intent=clamp_text(draft.behavioral_intent, 48),
        )
        system = PERSONA_POLISH_SYSTEM
        max_tokens = settings.programmatic_nlg_max_tokens
    else:
        user = f"CITIZEN BRIEF:\n{brief}\n\nWrite the JSON comment now."
        system = NATURALIZE_SYSTEM
        max_tokens = settings.programmatic_nlg_max_tokens

    if raw_comment and settings.persona_polish_openrouter and settings.openrouter_enabled:
        try:
            data = await get_openrouter().chat_json_persona(
                system, user, temperature=0.35
            )
            polished = str(data.get("comment", "")).strip()
            if polished and len(polished) >= 10:
                return polished[:220], "llm_polished"
        except Exception as exc:  # noqa: BLE001
            logger.debug("OpenRouter naturalize failed for %s: %s", draft.agent_id, exc)

    try:
        comment = await _ollama_naturalize(
            system, user, draft=draft, max_tokens=max_tokens
        )
        if comment:
            return comment, "naturalized" if not raw_comment else "llm_polished"
    except Exception as exc:  # noqa: BLE001
        logger.debug("Ollama naturalize failed for %s: %s", draft.agent_id, exc)

    return draft_fallback_line(draft), "draft_fallback"


async def _naturalize_one_with_fallback(
    draft: CognitiveDraft,
    *,
    max_tokens: int,
) -> tuple[str, str]:
    """Single brief: one Ollama call, then draft_fallback (no retry storm)."""
    user = f"CITIZEN BRIEF:\n{draft.to_naturalize_brief()}\n\nWrite the JSON comment now."
    try:
        comment = await _ollama_naturalize(
            NATURALIZE_SYSTEM, user, draft=draft, max_tokens=max_tokens
        )
        if comment:
            return comment, "naturalized"
    except Exception as exc:  # noqa: BLE001
        logger.debug("Single naturalize failed for %s: %s", draft.agent_id, exc)
    return draft_fallback_line(draft), "draft_fallback"


async def _naturalize_pack_with_fallback(
    drafts: list[CognitiveDraft],
    *,
    max_tokens: int,
) -> list[tuple[str, str]]:
    """Try packed Ollama call; on failure retry each draft once (no recursive split)."""
    if not drafts:
        return []
    if len(drafts) == 1:
        return [await _naturalize_one_with_fallback(drafts[0], max_tokens=max_tokens)]

    try:
        comments = await _ollama_naturalize_pack(drafts, max_tokens=max_tokens)
        if len(comments) == len(drafts) and all(comments):
            return [(c, "naturalized") for c in comments]
    except Exception as exc:  # noqa: BLE001
        logger.debug("Packed naturalize failed (%d drafts): %s", len(drafts), exc)

    return list(
        await asyncio.gather(
            *(_naturalize_one_with_fallback(d, max_tokens=max_tokens) for d in drafts)
        )
    )


async def naturalize_drafts_batch(
    drafts: list[CognitiveDraft],
    *,
    session_id: str | None = None,
    flow_uuid: str | None = None,
) -> tuple[list[tuple[str, str]], dict[str, Any]]:
    """Naturalize many drafts with packed Ollama calls; return (comment, source) pairs + stats."""
    settings = get_settings()
    if settings.programmatic_nlg_mode == "draft_only":
        results = [(draft_fallback_line(d), "draft_fallback") for d in drafts]
        stats = {
            "count": len(drafts),
            "elapsed_ms": 0,
            "naturalized": 0,
            "llm_polished": 0,
            "fallback_count": len(drafts),
            "pack_size": 0,
        }
        return results, stats

    pack_size = max(1, settings.naturalize_pack_size)
    max_tokens = settings.programmatic_nlg_max_tokens
    t0 = time.monotonic()
    chunks = [drafts[i : i + pack_size] for i in range(0, len(drafts), pack_size)]
    chunk_sem = asyncio.Semaphore(max(1, settings.ollama_concurrency))

    async def _run_chunk(chunk: list[CognitiveDraft]) -> list[tuple[str, str]]:
        async with chunk_sem:
            return await _naturalize_pack_with_fallback(chunk, max_tokens=max_tokens)

    chunk_results = await asyncio.gather(*(_run_chunk(chunk) for chunk in chunks))
    results = [pair for chunk in chunk_results for pair in chunk]
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    sources = [src for _, src in results]
    stats: dict[str, Any] = {
        "count": len(drafts),
        "elapsed_ms": elapsed_ms,
        "naturalized": sources.count("naturalized"),
        "llm_polished": sources.count("llm_polished"),
        "fallback_count": sources.count("draft_fallback"),
        "pack_size": pack_size,
        "pack_calls": len(chunks),
    }
    if stats["count"]:
        try:
            from observability.master_log import log_entry

            log_entry(
                "algo",
                "psychometric",
                "naturalize_complete",
                session_id=session_id,
                flow_uuid=flow_uuid,
                elapsed_ms=elapsed_ms,
                data=stats,
            )
        except Exception:  # noqa: BLE001
            pass
    logger.info(
        "Naturalize batch: %d agents in %dms (pack=%d calls=%d naturalized=%d fallback=%d)",
        stats["count"],
        elapsed_ms,
        pack_size,
        stats["pack_calls"],
        stats["naturalized"],
        stats["fallback_count"],
    )
    return results, stats
