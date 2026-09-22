"""OpenRouter chat-completions client.

Used for:
- Optional polish pass (`chat_json` / `chat_json_persona`) against OPENROUTER_MODEL
- Ollama fallback (`chat`) against a free-model cascade via OpenRouter's
  `model` + `models` array

If no API key is configured, callers should keep using local Gemma;
`enabled` / `api_available` expose that decision.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from config import get_settings

logger = logging.getLogger("singularity.openrouter")

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)

_HTTP: httpx.AsyncClient | None = None
_FALLBACK_SEM: asyncio.Semaphore | None = None


class OpenRouterError(RuntimeError):
    pass


def _fallback_semaphore() -> asyncio.Semaphore:
    global _FALLBACK_SEM
    if _FALLBACK_SEM is None:
        _FALLBACK_SEM = asyncio.Semaphore(
            max(1, get_settings().openrouter_fallback_concurrency)
        )
    return _FALLBACK_SEM


class OpenRouterClient:
    def __init__(self) -> None:
        self.settings = get_settings()

    @property
    def enabled(self) -> bool:
        return self.settings.openrouter_enabled and self.settings.use_openrouter_polish

    @property
    def api_available(self) -> bool:
        return bool(self.settings.openrouter_api_key)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.settings.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "Project Singularity",
        }

    async def _client(self) -> httpx.AsyncClient:
        global _HTTP
        if _HTTP is None or _HTTP.is_closed:
            _HTTP = httpx.AsyncClient(
                base_url=self.settings.openrouter_base_url.rstrip("/"),
                timeout=90.0,
            )
        return _HTTP

    async def _post_chat(self, body: dict[str, Any]) -> str:
        if not self.api_available:
            raise OpenRouterError("OpenRouter API key not configured")
        client = await self._client()
        try:
            resp = await client.post(
                "/chat/completions",
                headers=self._headers(),
                json=body,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:300]
            raise OpenRouterError(
                f"OpenRouter returned {exc.response.status_code}: {detail}"
            ) from exc
        except httpx.HTTPError as exc:
            raise OpenRouterError(f"OpenRouter request failed: {exc}") from exc
        data = resp.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise OpenRouterError(f"OpenRouter response missing content: {data!r}"[:300]) from exc
        return (content or "").strip()

    async def chat(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        json_mode: bool = False,
        models: list[str] | None = None,
    ) -> str:
        """Text (or JSON-string) completion with optional model cascade.

        `models` is the full preference list: first slug is `model`, the rest
        are OpenRouter server-side fallbacks.
        """
        chain = [m for m in (models or self.settings.openrouter_fallback_model_list) if m]
        if not chain:
            raise OpenRouterError("No OpenRouter fallback models configured")

        body: dict[str, Any] = {
            "model": chain[0],
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "provider": {"allow_fallbacks": True},
        }
        if len(chain) > 1:
            body["models"] = chain[1:]
        if max_tokens:
            body["max_tokens"] = max_tokens
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        async with _fallback_semaphore():
            return await self._post_chat(body)

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=8), reraise=True)
    async def chat_json(self, system: str, user: str, *, temperature: float = 0.5) -> dict[str, Any]:
        """Report polish — gated by USE_OPENROUTER_POLISH via `enabled` at the caller."""
        content = await self._post_chat(self._polish_body(system, user, temperature=temperature))
        return _parse_json(content)

    async def chat_json_persona(
        self, system: str, user: str, *, temperature: float = 0.35
    ) -> dict[str, Any]:
        """Persona polish — uses API key only (not USE_OPENROUTER_POLISH gate)."""
        content = await self._post_chat(self._polish_body(system, user, temperature=temperature))
        return _parse_json(content)

    def _polish_body(self, system: str, user: str, *, temperature: float) -> dict[str, Any]:
        return {
            "model": self.settings.openrouter_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }


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


def reset_openrouter_runtime() -> None:
    """Drop singleton client / HTTP pool (tests)."""
    global _client, _HTTP, _FALLBACK_SEM
    _client = None
    if _HTTP is not None and not _HTTP.is_closed:
        # Tests run in a single event loop; closing here is safe after awaits.
        _HTTP = None
    else:
        _HTTP = None
    _FALLBACK_SEM = None
