"""Append-only global master log (JSON Lines) for flow, algo, data, and client events."""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import get_settings

_write_lock = threading.Lock()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_log_path() -> Path:
    settings = get_settings()
    path = Path(settings.master_log_path)
    if not path.is_absolute():
        path = repo_root() / path
    return path


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def state_snapshot(state: Any, phase: str | None = None) -> dict[str, Any]:
    """Compact counts from SingularityState — never full persona rows."""
    dag_nodes = len(state.dag.nodes) if state.dag else 0
    causal_nodes = len(state.causal.nodes) if state.causal else 0
    forecast_model = state.forecast.model if state.forecast else None
    forecast_horizon = state.forecast.horizon_days if state.forecast else None
    snap: dict[str, Any] = {
        "phase": phase,
        "evidence_count": len(state.evidence),
        "series_count": len(state.series),
        "persona_responses": len(state.persona_responses),
        "persona_opinions": len(state.persona_opinions),
        "personas_simulated": state.metrics.get("personas", len(state.persona_opinions)),
        "dag_nodes": dag_nodes,
        "nodes_resolved": None,
        "metrics_keys": sorted(state.metrics.keys()),
        "forecast_model": forecast_model,
        "forecast_horizon_days": forecast_horizon,
        "causal_nodes": causal_nodes,
        "report_sections": len(state.report_sections),
    }
    return snap


def phase_result_summary(state: Any, phase: str) -> dict[str, Any]:
    """Algo-specific metrics already on state.metrics after a phase completes."""
    m = state.metrics
    if phase == "prediction_market":
        pm = m.get("prediction_market") or {}
        return {"overall_outcome": pm.get("overall_outcome")}
    if phase == "monte_carlo":
        mc = m.get("monte_carlo") or {}
        return {
            "most_likely": mc.get("most_likely"),
            "n_simulations": mc.get("n_simulations"),
        }
    if phase == "swarm_optimization":
        sw = m.get("swarm_optimization") or {}
        return {
            "domain": sw.get("domain"),
            "convergence_iterations": sw.get("convergence_iterations"),
        }
    if phase == "decision_engine":
        de = m.get("decision_engine") or {}
        opts = de.get("options") or []
        return {"option_count": len(opts) if isinstance(opts, list) else 0}
    if phase == "causal":
        return {"causal_nodes": len(state.causal.nodes) if state.causal else 0}
    if phase == "decompose_dag":
        return {"dag_nodes": len(state.dag.nodes) if state.dag else 0}
    return {}


def log_entry(
    source: str,
    category: str,
    event: str,
    *,
    session_id: str | None = None,
    flow_uuid: str | None = None,
    phase: str | None = None,
    elapsed_ms: int | None = None,
    data: dict[str, Any] | None = None,
) -> None:
    settings = get_settings()
    if not settings.master_log_enabled:
        return

    record: dict[str, Any] = {
        "ts": _utc_now_iso(),
        "source": source,
        "category": category,
        "event": event,
    }
    if session_id is not None:
        record["session_id"] = session_id
    if flow_uuid is not None:
        record["flow_uuid"] = flow_uuid
    if phase is not None:
        record["phase"] = phase
    if elapsed_ms is not None:
        record["elapsed_ms"] = elapsed_ms
    if data:
        record["data"] = data

    path = resolve_log_path()
    line = json.dumps(record, default=str, ensure_ascii=False) + "\n"
    with _write_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line)


def log_flow(state: Any, event: str, *, data: dict[str, Any] | None = None, elapsed_ms: int | None = None) -> None:
    log_entry(
        "backend",
        "flow",
        event,
        session_id=state.session_id or state.flow_uuid,
        flow_uuid=state.flow_uuid,
        elapsed_ms=elapsed_ms,
        data=data,
    )


def log_algo(state: Any, event: str, phase: str, *, elapsed_ms: int | None = None, data: dict[str, Any] | None = None) -> None:
    log_entry(
        "backend",
        "algo",
        event,
        session_id=state.session_id or state.flow_uuid,
        flow_uuid=state.flow_uuid,
        phase=phase,
        elapsed_ms=elapsed_ms,
        data=data,
    )


def log_heartbeat(state: Any, phase: str, elapsed_ms: int) -> None:
    snap = state_snapshot(state, phase)
    log_entry(
        "backend",
        "heartbeat",
        "heartbeat",
        session_id=state.session_id or state.flow_uuid,
        flow_uuid=state.flow_uuid,
        phase=phase,
        elapsed_ms=elapsed_ms,
        data=snap,
    )


def log_data(event: str, *, session_id: str | None = None, flow_uuid: str | None = None, data: dict[str, Any] | None = None) -> None:
    log_entry(
        "data",
        "store",
        event,
        session_id=session_id,
        flow_uuid=flow_uuid,
        data=data,
    )
