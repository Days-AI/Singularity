"""Async Ollama client for local Gemma inference.

Talks to the Ollama HTTP API (/api/generate). Supports JSON-constrained
generation (format=json) for the DAG decomposer and persona simulator, with
defensive JSON extraction since small local models occasionally wrap output.

A process-wide concurrency limit prevents saturating Ollama when the
psychometric engine fans out dozens of archetype calls in parallel.

When Ollama is unreachable or errors and OpenRouter is configured, generate()
falls back to a free-model cascade (see OPENROUTER_FALLBACK_MODELS).
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any

import httpx
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_exponential

from config import get_settings
from llm.openrouter_client import OpenRouterError, get_openrouter
from nlp.prompt_budget import fit_prompt_pair

logger = logging.getLogger("singularity.ollama")

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)
JSON_PREFERRED_FALLBACK_MODEL = "openai/gpt-oss-20b:free"

# Bounds in-flight requests so we don't saturate Ollama when the psychometric
# engine fans out dozens of archetype calls. Built lazily from settings so it
# binds to the running event loop and respects OLLAMA_CONCURRENCY.
_OLLAMA_SEM: asyncio.Semaphore | None = None
_HTTP: httpx.AsyncClient | None = None
_MODEL_OK: bool | None = None

_CIRCUIT_FAILURES = 0
_CIRCUIT_OPEN_UNTIL = 0.0


def _semaphore() -> asyncio.Semaphore:
    global _OLLAMA_SEM
    if _OLLAMA_SEM is None:
        _OLLAMA_SEM = asyncio.Semaphore(max(1, get_settings().ollama_concurrency))
    return _OLLAMA_SEM


class OllamaError(RuntimeError):
    pass


class OllamaUnreachableError(OllamaError):
    """Connect/timeout — do not retry locally; trip the circuit and fall back."""


def fallback_model_chain(*, json_mode: bool) -> list[str]:
    """OpenRouter preference list; json_mode puts gpt-oss-20b first for structured output."""
    models = get_settings().openrouter_fallback_model_list
    if json_mode and JSON_PREFERRED_FALLBACK_MODEL in models:
        return [JSON_PREFERRED_FALLBACK_MODEL, *[m for m in models if m != JSON_PREFERRED_FALLBACK_MODEL]]
    return list(models)


def _circuit_open() -> bool:
    global _CIRCUIT_FAILURES, _CIRCUIT_OPEN_UNTIL
    if _CIRCUIT_OPEN_UNTIL <= 0:
        return False
    if time.monotonic() < _CIRCUIT_OPEN_UNTIL:
        return True
    _CIRCUIT_FAILURES = 0
    _CIRCUIT_OPEN_UNTIL = 0.0
    return False


def _circuit_record_success() -> None:
    global _CIRCUIT_FAILURES, _CIRCUIT_OPEN_UNTIL
    _CIRCUIT_FAILURES = 0
    _CIRCUIT_OPEN_UNTIL = 0.0


def _circuit_record_failure() -> None:
    global _CIRCUIT_FAILURES, _CIRCUIT_OPEN_UNTIL
    settings = get_settings()
    _CIRCUIT_FAILURES += 1
    threshold = max(1, settings.ollama_circuit_failures)
    if _CIRCUIT_FAILURES >= threshold:
        _CIRCUIT_OPEN_UNTIL = time.monotonic() + settings.ollama_circuit_cooldown_s
        logger.warning(
            "Ollama circuit open for %.0fs after %d failures; using OpenRouter fallback",
            settings.ollama_circuit_cooldown_s,
            _CIRCUIT_FAILURES,
        )


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
        try:
            models = await self.list_models()
        except httpx.ConnectError as exc:
            raise OllamaUnreachableError(
                f"Cannot reach Ollama at {self.base_url}. Is `ollama serve` running?"
            ) from exc
        except httpx.TimeoutException as exc:
            raise OllamaUnreachableError(
                f"Ollama timed out at {self.base_url}"
            ) from exc
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

    async def generate(
        self,
        system: str,
        prompt: str,
        *,
        temperature: float = 0.7,
        top_p: float | None = None,
        repeat_penalty: float | None = None,
        json_mode: bool = False,
        max_tokens: int | None = None,
        num_ctx: int | None = None,
    ) -> str:
        use_fallback = self.settings.ollama_fallback_enabled
        if not _circuit_open():
            try:
                text = await self._generate_local(
                    system,
                    prompt,
                    temperature=temperature,
                    top_p=top_p,
                    repeat_penalty=repeat_penalty,
                    json_mode=json_mode,
                    max_tokens=max_tokens,
                    num_ctx=num_ctx,
                )
                _circuit_record_success()
                return text
            except OllamaError as exc:
                _circuit_record_failure()
                if not use_fallback:
                    raise
                logger.warning("Ollama failed; falling back to OpenRouter: %s", exc)
        elif not use_fallback:
            raise OllamaError("Ollama circuit is open and OpenRouter fallback is disabled")
        else:
            logger.info("Ollama circuit open; using OpenRouter fallback")

        return await self._generate_openrouter(
            system,
            prompt,
            temperature=temperature,
            json_mode=json_mode,
            max_tokens=max_tokens,
        )

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(min=1, max=6),
        retry=retry_if_not_exception_type(OllamaUnreachableError),
        reraise=True,
    )
    async def _generate_local(
        self,
        system: str,
        prompt: str,
        *,
        temperature: float = 0.7,
        top_p: float | None = None,
        repeat_penalty: float | None = None,
        json_mode: bool = False,
        max_tokens: int | None = None,
        num_ctx: int | None = None,
    ) -> str:
        await self.ensure_model()
        ctx = num_ctx if num_ctx is not None else self.settings.ollama_num_ctx
        payload: dict[str, Any] = {
            "model": self.model,
            "system": system,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_ctx": ctx,
            },
        }
        if top_p is not None:
            payload["options"]["top_p"] = top_p
        if repeat_penalty is not None:
            payload["options"]["repeat_penalty"] = repeat_penalty
        if json_mode:
            payload["format"] = "json"
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        predict = max_tokens or 256
        system, prompt, trimmed = fit_prompt_pair(
            system,
            prompt,
            num_ctx=ctx,
            num_predict=predict,
        )
        payload["system"] = system
        payload["prompt"] = prompt
        if trimmed:
            logger.warning(
                "Trimmed Ollama prompt to fit num_ctx=%d (num_predict=%d)",
                ctx,
                predict,
            )

        async with _semaphore():
            client = await self._client()
            try:
                resp = await client.post("/api/generate", json=payload)
                resp.raise_for_status()
            except httpx.ConnectError as exc:
                raise OllamaUnreachableError(
                    f"Cannot reach Ollama at {self.base_url}. Is `ollama serve` running?"
                ) from exc
            except httpx.TimeoutException as exc:
                raise OllamaUnreachableError(
                    f"Ollama timed out at {self.base_url}"
                ) from exc
            except httpx.HTTPStatusError as exc:
                detail = exc.response.text[:300]
                raise OllamaError(
                    f"Ollama returned {exc.response.status_code} for model '{self.model}': {detail}"
                ) from exc
            data = resp.json()
        return (data.get("response") or "").strip()

    async def _generate_openrouter(
        self,
        system: str,
        prompt: str,
        *,
        temperature: float,
        json_mode: bool,
        max_tokens: int | None,
    ) -> str:
        chain = fallback_model_chain(json_mode=json_mode)
        try:
            return await get_openrouter().chat(
                system,
                prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=json_mode,
                models=chain,
            )
        except OpenRouterError as exc:
            raise OllamaError(f"Ollama failed and OpenRouter fallback failed: {exc}") from exc

    async def generate_json(
        self,
        system: str,
        prompt: str,
        *,
        temperature: float = 0.4,
        top_p: float | None = None,
        repeat_penalty: float | None = None,
        max_tokens: int | None = None,
        num_ctx: int | None = None,
    ) -> dict[str, Any]:
        """Generate and parse a JSON object, tolerating minor wrapping.

        `max_tokens` caps `num_predict` so structured callers terminate
        generation early instead of letting the model ramble.
        """
        raw = await self.generate(
            system,
            prompt,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            json_mode=True,
            max_tokens=max_tokens,
            num_ctx=num_ctx,
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
            "fallback": {
                "enabled": self.settings.ollama_fallback_enabled,
                "models": self.settings.openrouter_fallback_model_list,
                "circuit_open": _circuit_open(),
            },
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


def reset_ollama_runtime() -> None:
    """Drop singleton client, HTTP pool, and circuit state (tests)."""
    global _client, _HTTP, _OLLAMA_SEM, _MODEL_OK, _CIRCUIT_FAILURES, _CIRCUIT_OPEN_UNTIL
    _client = None
    _HTTP = None
    _OLLAMA_SEM = None
    _MODEL_OK = None
    _CIRCUIT_FAILURES = 0
    _CIRCUIT_OPEN_UNTIL = 0.0
