"""Async Ollama client for local Gemma inference.

Talks to the Ollama HTTP API (/api/generate). Supports JSON-constrained
generation (format=json) for the DAG decomposer and persona simulator, with
defensive JSON extraction since small local models occasionally wrap output.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from config import get_settings

logger = logging.getLogger("singularity.ollama")

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.base_url = self.settings.ollama_base_url.rstrip("/")
        self.model = self.settings.ollama_model

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=6), reraise=True)
    async def generate(
        self,
        system: str,
        prompt: str,
        *,
        temperature: float = 0.7,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "system": system,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_ctx": self.settings.ollama_num_ctx,
            },
        }
        if json_mode:
            payload["format"] = "json"
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        async with httpx.AsyncClient(timeout=self.settings.ollama_timeout_s) as client:
            resp = await client.post(f"{self.base_url}/api/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()
        return (data.get("response") or "").strip()

    async def generate_json(
        self, system: str, prompt: str, *, temperature: float = 0.4
    ) -> dict[str, Any]:
        """Generate and parse a JSON object, tolerating minor wrapping."""
        raw = await self.generate(system, prompt, temperature=temperature, json_mode=True)
        return _parse_json(raw)

    async def ping(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                return resp.status_code == 200
        except Exception:  # noqa: BLE001
            return False


def _parse_json(raw: str) -> dict[str, Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = _JSON_BLOCK.search(raw)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError as exc:
                raise OllamaError(f"Could not parse JSON from model output: {raw[:200]}") from exc
        raise OllamaError(f"No JSON object in model output: {raw[:200]}")


_client: OllamaClient | None = None


def get_ollama() -> OllamaClient:
    global _client
    if _client is None:
        _client = OllamaClient()
    return _client
