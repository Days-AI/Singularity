from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import pytest

from config import get_settings
from observability.master_log import log_entry, log_heartbeat, resolve_log_path, state_snapshot
from state import SingularityState

_TEST_LOG_DIR = Path(__file__).resolve().parent / "_master_log_tmp"


@pytest.fixture
def master_log_path(monkeypatch: pytest.MonkeyPatch) -> Path:
    if _TEST_LOG_DIR.exists():
        shutil.rmtree(_TEST_LOG_DIR, ignore_errors=True)
    _TEST_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = _TEST_LOG_DIR / f"test_{uuid.uuid4().hex}.jsonl"
    monkeypatch.setenv("MASTER_LOG_PATH", str(log_file))
    monkeypatch.setenv("MASTER_LOG_ENABLED", "true")
    get_settings.cache_clear()
    yield log_file
    get_settings.cache_clear()
    if _TEST_LOG_DIR.exists():
        shutil.rmtree(_TEST_LOG_DIR, ignore_errors=True)


def test_log_entry_writes_jsonl(master_log_path: Path) -> None:
    log_entry(
        "backend",
        "flow",
        "test_event",
        session_id="sess-1",
        flow_uuid="flow-1",
        data={"ok": True},
    )
    lines = master_log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["source"] == "backend"
    assert row["event"] == "test_event"
    assert row["session_id"] == "sess-1"
    assert row["data"]["ok"] is True


def test_state_snapshot_keys() -> None:
    state = SingularityState(query="test query", flow_uuid="abc")
    snap = state_snapshot(state, "psychometric")
    assert snap["phase"] == "psychometric"
    assert snap["evidence_count"] == 0
    assert snap["persona_opinions"] == 0
    assert "metrics_keys" in snap


def test_resolve_log_path_relative_to_repo(master_log_path: Path) -> None:
    assert resolve_log_path() == master_log_path


def test_client_log_endpoint(master_log_path: Path) -> None:
    from main import app
    from fastapi.testclient import TestClient

    client = TestClient(app)
    res = client.post(
        "/api/log",
        json={
            "event": "stream_connect",
            "session_id": "s1",
            "category": "client",
            "data": {"connection": "streaming"},
        },
    )
    assert res.status_code == 200
    row = json.loads(master_log_path.read_text(encoding="utf-8").strip())
    assert row["source"] == "frontend"
    assert row["event"] == "stream_connect"


def test_heartbeat_entry_shape(master_log_path: Path) -> None:
    state = SingularityState(query="q", flow_uuid="f", session_id="s")
    log_heartbeat(state, "evidence", 60000)
    row = json.loads(master_log_path.read_text(encoding="utf-8").strip())
    assert row["category"] == "heartbeat"
    assert row["phase"] == "evidence"
    assert row["elapsed_ms"] == 60000
    assert row["data"]["evidence_count"] == 0
