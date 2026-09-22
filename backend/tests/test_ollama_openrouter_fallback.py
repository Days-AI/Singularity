"""Ollama → OpenRouter free-model fallback."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from config import clear_settings_cache, get_settings
from llm.ollama_client import (
    JSON_PREFERRED_FALLBACK_MODEL,
    OllamaUnreachableError,
    fallback_model_chain,
    get_ollama,
    reset_ollama_runtime,
)
from llm.openrouter_client import get_openrouter, reset_openrouter_runtime

NEMOTRON = "nvidia/nemotron-3-ultra-550b-a55b:free"
GEMMA_31 = "google/gemma-4-31b-it:free"
GEMMA_26 = "google/gemma-4-26b-a4b-it:free"
GPT_OSS = "openai/gpt-oss-20b:free"


@pytest.fixture(autouse=True)
def _runtime(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OLLAMA_OPENROUTER_FALLBACK", "true")
    monkeypatch.setenv("OPENROUTER_MODEL", "google/gemma-2-27b-it")
    clear_settings_cache()
    reset_ollama_runtime()
    reset_openrouter_runtime()
    yield
    reset_ollama_runtime()
    reset_openrouter_runtime()
    clear_settings_cache()


def test_fallback_model_chain_default_order():
    chain = fallback_model_chain(json_mode=False)
    assert chain[0] == NEMOTRON
    assert GEMMA_31 in chain
    assert GEMMA_26 in chain
    assert chain[-1] == GPT_OSS


def test_fallback_model_chain_json_prefers_gpt_oss():
    chain = fallback_model_chain(json_mode=True)
    assert chain[0] == JSON_PREFERRED_FALLBACK_MODEL
    assert chain[0] == GPT_OSS
    assert NEMOTRON in chain[1:]


def test_settings_fallback_helpers():
    s = get_settings()
    assert s.ollama_fallback_enabled is True
    assert s.openrouter_fallback_model_list[0] == NEMOTRON


@pytest.mark.asyncio
async def test_ollama_unreachable_falls_back_to_openrouter_chain():
    client = get_ollama()
    local = AsyncMock(side_effect=OllamaUnreachableError("Cannot reach Ollama"))
    with (
        patch.object(client, "_generate_local", local),
        patch("llm.ollama_client.get_openrouter") as gor,
    ):
        gor.return_value.chat = AsyncMock(return_value="cloud text")
        text = await client.generate("sys", "prompt")
    assert text == "cloud text"
    models = gor.return_value.chat.call_args.kwargs["models"]
    assert models[0] == NEMOTRON
    assert models[1] == GEMMA_31
    assert models[2] == GEMMA_26
    assert models[3] == GPT_OSS


@pytest.mark.asyncio
async def test_json_mode_puts_gpt_oss_first():
    client = get_ollama()
    local = AsyncMock(side_effect=OllamaUnreachableError("Cannot reach Ollama"))
    with (
        patch.object(client, "_generate_local", local),
        patch("llm.ollama_client.get_openrouter") as gor,
    ):
        gor.return_value.chat = AsyncMock(return_value='{"ok": true}')
        text = await client.generate("sys", "prompt", json_mode=True)
    assert text == '{"ok": true}'
    models = gor.return_value.chat.call_args.kwargs["models"]
    assert models[0] == GPT_OSS
    assert gor.return_value.chat.call_args.kwargs["json_mode"] is True


@pytest.mark.asyncio
async def test_no_api_key_raises_original_error_without_openrouter():
    client = get_ollama()
    client.settings.openrouter_api_key = None
    local = AsyncMock(side_effect=OllamaUnreachableError("Cannot reach Ollama at localhost"))
    with (
        patch.object(client, "_generate_local", local),
        patch("llm.ollama_client.get_openrouter") as gor,
    ):
        with pytest.raises(OllamaUnreachableError, match="Cannot reach Ollama"):
            await client.generate("sys", "prompt")
        gor.assert_not_called()


@pytest.mark.asyncio
async def test_circuit_skips_ollama_after_threshold_failures():
    client = get_ollama()
    local = AsyncMock(side_effect=OllamaUnreachableError("Cannot reach Ollama"))
    with (
        patch.object(client, "_generate_local", local),
        patch("llm.ollama_client.get_openrouter") as gor,
    ):
        gor.return_value.chat = AsyncMock(return_value="cloud")
        await client.generate("sys", "p1")
        await client.generate("sys", "p2")
        assert local.call_count == 2
        await client.generate("sys", "p3")
        assert local.call_count == 2
    assert gor.return_value.chat.call_count == 3


@pytest.mark.asyncio
async def test_local_success_does_not_call_openrouter():
    client = get_ollama()
    with (
        patch.object(client, "_generate_local", AsyncMock(return_value="local gemma")),
        patch("llm.ollama_client.get_openrouter") as gor,
    ):
        text = await client.generate("sys", "prompt")
    assert text == "local gemma"
    gor.assert_not_called()


@pytest.mark.asyncio
async def test_status_includes_fallback_state():
    client = get_ollama()
    with patch.object(client, "ping", AsyncMock(return_value=False)):
        st = await client.status()
    assert st["reachable"] is False
    assert st["fallback"]["enabled"] is True
    assert NEMOTRON in st["fallback"]["models"]
    assert st["fallback"]["circuit_open"] is False


@pytest.mark.asyncio
async def test_openrouter_chat_sends_model_and_models_array():
    captured: dict = {}

    class FakeResp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"choices": [{"message": {"content": "hello"}}]}

    class FakeHttp:
        async def post(self, path: str, headers=None, json=None):
            captured["path"] = path
            captured["body"] = json
            return FakeResp()

    or_client = get_openrouter()
    with patch.object(or_client, "_client", AsyncMock(return_value=FakeHttp())):
        text = await or_client.chat(
            "sys",
            "user",
            models=[NEMOTRON, GEMMA_31, GPT_OSS],
        )
    assert text == "hello"
    assert captured["path"] == "/chat/completions"
    assert captured["body"]["model"] == NEMOTRON
    assert captured["body"]["models"] == [GEMMA_31, GPT_OSS]
    assert captured["body"]["provider"]["allow_fallbacks"] is True


@pytest.mark.asyncio
async def test_polish_chat_json_uses_openrouter_model_only():
    captured: dict = {}

    class FakeResp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"choices": [{"message": {"content": '{"ok": true}'}}]}

    class FakeHttp:
        async def post(self, path: str, headers=None, json=None):
            captured["path"] = path
            captured["body"] = json
            return FakeResp()

    or_client = get_openrouter()
    with patch.object(or_client, "_client", AsyncMock(return_value=FakeHttp())):
        data = await or_client.chat_json("sys", "user")
    assert data == {"ok": True}
    assert captured["body"]["model"] == "google/gemma-2-27b-it"
    assert "models" not in captured["body"]
