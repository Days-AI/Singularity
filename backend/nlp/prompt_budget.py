"""Trim LLM prompts so system+user fit within Ollama num_ctx."""
from __future__ import annotations

import re

# Rough chars-per-token for English prose (conservative for Gemma).
_CHARS_PER_TOKEN = 3.5


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def clamp_text(text: str, max_chars: int, suffix: str = "…") -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    cut = max(0, max_chars - len(suffix))
    return text[:cut].rstrip() + suffix


def compact_debate_lines(lines: list[str], max_lines: int = 3, max_line_chars: int = 48) -> str:
    if not lines:
        return "minimal internal conflict"
    picked = lines[:max_lines]
    return "; ".join(clamp_text(line, max_line_chars, "") for line in picked)


def fit_prompt_pair(
    system: str,
    user: str,
    *,
    num_ctx: int,
    num_predict: int,
    reserve_tokens: int = 64,
) -> tuple[str, str, bool]:
    """Truncate user (then system if needed) to fit context window."""
    budget = num_ctx - num_predict - reserve_tokens
    if budget < 256:
        budget = 256
    sys_t = estimate_tokens(system)
    user_t = estimate_tokens(user)
    if sys_t + user_t <= budget:
        return system, user, False

    trimmed = False
    user_budget = budget - sys_t
    if user_budget < 160 and sys_t > budget // 2:
        system = clamp_text(system, int(budget * 0.4 * _CHARS_PER_TOKEN))
        sys_t = estimate_tokens(system)
        user_budget = budget - sys_t
        trimmed = True

    user_max = int(max(128, user_budget) * _CHARS_PER_TOKEN)
    if len(user) > user_max:
        user = clamp_text(user, user_max)
        trimmed = True
    return system, user, trimmed


def trim_json_payload(payload: object, max_chars: int = 4500) -> str:
    """Serialize JSON for LLM prompts, dropping heavy fields if over budget."""
    import json

    text = json.dumps(payload, default=str)
    if len(text) <= max_chars:
        return text
    if isinstance(payload, dict):
        slim = dict(payload)
        for key in ("rag_context", "web_intelligence", "deliberation", "raw_comments"):
            if key not in slim:
                continue
            val = slim[key]
            if isinstance(val, str):
                slim[key] = clamp_text(val, 400)
            elif isinstance(val, dict):
                slim[key] = dict(list(val.items())[:4])
            elif isinstance(val, list):
                slim[key] = val[:3]
        text = json.dumps(slim, default=str)
        if len(text) <= max_chars:
            return text
    return clamp_text(text, max_chars)


def compact_nlg_context(
    query: str,
    *,
    keyphrases: list[str] | None = None,
    evidence_hook: str = "",
    max_query_chars: int = 160,
    max_hook_chars: int = 140,
) -> tuple[str, str]:
    """Return (stimulus, context_blurb) without duplicating long evidence lists."""
    stimulus = clamp_text(query, max_query_chars)
    if keyphrases:
        kp = clamp_text(keyphrases[0], 60, "")
        stimulus = clamp_text(f"{stimulus} — focus: {kp}", max_query_chars + 40)
    if evidence_hook:
        context = clamp_text(evidence_hook, max_hook_chars)
    elif keyphrases and len(keyphrases) > 1:
        context = clamp_text(", ".join(keyphrases[1:3]), max_hook_chars)
    else:
        context = stimulus
    return stimulus, context
