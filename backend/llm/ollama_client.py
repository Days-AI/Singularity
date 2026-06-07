"""Async Ollama client for local Gemma inference.

Talks to the Ollama HTTP API (/api/generate). Supports JSON-constrained
generation (format=json) for the DAG decomposer and persona simulator, with
defensive JSON extraction since small local models occasionally wrap output.

A process-wide concurrency limit prevents saturating Ollama when the
psychometric engine fans out dozens of archetype calls in parallel.
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

logger = logging.getLogger("singularity.ollama")

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)

# Bounds in-flight requests so we don't saturate Ollama when the psychometric
# engine fans out dozens of archetype calls. Built lazily from settings so it
# binds to the running event loop and respects OLLAMA_CONCURRENCY.
_OLLAMA_SEM: asyncio.Semaphore | None = None
_HTTP: httpx.AsyncClient | None = None
_MODEL_OK: bool | None = None


def _semaphore() -> asyncio.Semaphore:
    global _OLLAMA_SEM
    if _OLLAMA_SEM is None:
        _OLLAMA_SEM = asyncio.Semaphore(max(1, get_settings().ollama_concurrency))
    return _OLLAMA_SEM


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.base_url = self.settings.ollama_base_url.rstrip("/")
        self.model = self.settings.ollama_model

    async def _client(self) -> httpx.AsyncClient:
        global _HTTP
        if _HTTP is None or _HTTP.is_closed:
            _HTTP = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.settings.ollama_timeout_s,
            )
        return _HTTP

    async def list_models(self) -> list[str]:
        client = await self._client()
        resp = await client.get("/api/tags")
        resp.raise_for_status()
        data = resp.json()
        return [m.get("name", "") for m in data.get("models", []) if m.get("name")]

    async def ensure_model(self) -> None:
        """Verify the configured model tag exists locally; raise if missing."""
        global _MODEL_OK
        if _MODEL_OK is True:
            return
        models = await self.list_models()
        if not models:
            raise OllamaError(
                f"No models found at {self.base_url}. Run: ollama pull {self.model}"
            )
        wanted = self.model
        if wanted in models:
            _MODEL_OK = True
            return
        # Accept partial tag match (e.g. gemma4:latest vs gemma4:e4b).
        base = wanted.split(":")[0]
        matches = [m for m in models if m == wanted or m.startswith(f"{base}:")]
        if matches:
            if matches[0] != wanted:
                logger.info("Ollama model %s not found; using %s", wanted, matches[0])
                self.model = matches[0]
            _MODEL_OK = True
            return
        raise OllamaError(
            f"Model '{wanted}' not found in Ollama. Available: {', '.join(models)}. "
            f"Run: ollama pull {wanted}"
        )

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
        await self.ensure_model()
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

        async with _semaphore():
            client = await self._client()
            try:
                resp = await client.post("/api/generate", json=payload)
                resp.raise_for_status()
            except httpx.ConnectError as exc:
                raise OllamaError(
                    f"Cannot reach Ollama at {self.base_url}. Is `ollama serve` running?"
                ) from exc
            except httpx.HTTPStatusError as exc:
                detail = exc.response.text[:300]
                raise OllamaError(
                    f"Ollama returned {exc.response.status_code} for model '{self.model}': {detail}"
                ) from exc
            data = resp.json()
        return (data.get("response") or "").strip()

    async def generate_json(
        self,
        system: str,
        prompt: str,
        *,
        temperature: float = 0.4,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Generate and parse a JSON object, tolerating minor wrapping.

        `max_tokens` caps `num_predict` so structured callers terminate
        generation early instead of letting the model ramble.
        """
        raw = await self.generate(
            system, prompt, temperature=temperature, json_mode=True, max_tokens=max_tokens
        )
        return _parse_json(raw)

    async def ping(self) -> bool:
        try:
            client = await self._client()
            resp = await client.get("/api/tags", timeout=5.0)
            return resp.status_code == 200
        except Exception:  # noqa: BLE001
            return False

    async def status(self) -> dict[str, Any]:
        """Health snapshot for /api/health and startup logs."""
        reachable = await self.ping()
        out: dict[str, Any] = {
            "reachable": reachable,
            "base_url": self.base_url,
            "configured_model": self.settings.ollama_model,
            "active_model": self.model,
            "model_available": False,
            "available_models": [],
        }
        if not reachable:
            return out
        try:
            models = await self.list_models()
            out["available_models"] = models
            wanted = self.settings.ollama_model
            out["model_available"] = wanted in models or any(
                m.startswith(wanted.split(":")[0] + ":") for m in models
            )
        except Exception as exc:  # noqa: BLE001
            out["error"] = str(exc)
        return out


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
