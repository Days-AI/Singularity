"""OpenRouter polishing-layer client.

Wraps the OpenRouter chat-completions API for the McKinsey-grade prose pass
(PT-03). If no API key is configured, callers should fall back to local Gemma;
`enabled` exposes that decision.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from config import get_settings

logger = logging.getLogger("singularity.openrouter")

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


class OpenRouterClient:
    def __init__(self) -> None:
        self.settings = get_settings()

    @property
    def enabled(self) -> bool:
        return self.settings.openrouter_enabled

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=8), reraise=True)
    async def chat_json(self, system: str, user: str, *, temperature: float = 0.5) -> dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("OpenRouter not configured")

        headers = {
            "Authorization": f"Bearer {self.settings.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "Project Singularity",
        }
        body = {
            "model": self.settings.openrouter_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }
        async with httpx.AsyncClient(timeout=90.0) as client:
            resp = await client.post(
                f"{self.settings.openrouter_base_url}/chat/completions",
                headers=headers,
                json=body,
            )
            resp.raise_for_status()
            data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return _parse_json(content)


def _parse_json(raw: str) -> dict[str, Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = _JSON_BLOCK.search(raw)
        if match:
            return json.loads(match.group(0))
        raise


_client: OpenRouterClient | None = None


def get_openrouter() -> OpenRouterClient:
    global _client
    if _client is None:
        _client = OpenRouterClient()
    return _client
