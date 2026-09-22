"""Tests for Ollama prompt budgeting."""
from nlp.prompt_budget import compact_debate_lines, fit_prompt_pair, trim_json_payload


def test_fit_prompt_pair_trims_oversized_user():
    system = "You are a helper."
    user = "x" * 20000
    sys_out, user_out, trimmed = fit_prompt_pair(
        system, user, num_ctx=4096, num_predict=256
    )
    assert trimmed is True
    assert sys_out == system
    assert len(user_out) < len(user)


def test_fit_prompt_pair_keeps_small_prompts():
    system = "sys"
    user = "short user"
    _, user_out, trimmed = fit_prompt_pair(system, user, num_ctx=4096, num_predict=256)
    assert trimmed is False
    assert user_out == user


def test_compact_debate_lines():
    lines = ["Price Sensitivity: \"+0.42\" (w=0.91)", "Trust: \"-0.12\" (w=0.55)"]
    out = compact_debate_lines(lines)
    assert "Price" in out
    assert len(out) < 200


def test_trim_json_payload_drops_heavy_fields():
    payload = {
        "query": "test",
        "rag_context": "r" * 8000,
        "web_intelligence": {"items": list(range(50))},
    }
    text = trim_json_payload(payload, max_chars=2000)
    assert len(text) <= 2000
    assert "query" in text
