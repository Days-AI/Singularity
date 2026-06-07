"""Persistence layer.

A single `Store` abstraction backs the API's durable reads/writes. When
Supabase is configured it persists to Postgres; otherwise it transparently
falls back to an in-process store so the backend is fully functional offline.

All methods are defensive: a Supabase outage logs and degrades to memory rather
than failing the flow.
"""
from __future__ import annotations

import logging
import uuid
from functools import lru_cache
from typing import Any

from config import get_settings

logger = logging.getLogger("singularity.store")


class Store:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._client = None
        self._mem_sessions: dict[str, dict[str, Any]] = {}
        self._mem_reports: dict[str, dict[str, Any]] = {}
        if self.settings.supabase_enabled:
            self._init_supabase()

    def _init_supabase(self) -> None:
        try:
            from supabase import create_client

            self._client = create_client(
                self.settings.supabase_url, self.settings.supabase_service_key
            )
            logger.info("Supabase persistence enabled")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Supabase init failed, using in-memory store: %s", exc)
            self._client = None

    @property
    def enabled(self) -> bool:
        return self._client is not None

    async def create_session(self, query: str, flow_uuid: str) -> str:
        session_id = str(uuid.uuid4())
        row = {
            "id": session_id,
            "query": query,
            "flow_uuid": flow_uuid,
            "status": "pending",
        }
        self._mem_sessions[session_id] = row
        if self._client is not None:
            try:
                self._client.table("simulation_sessions").insert(
                    {"id": session_id, "query": query, "flow_uuid": flow_uuid, "status": "pending"}
                ).execute()
            except Exception as exc:  # noqa: BLE001
                logger.warning("create_session persist failed: %s", exc)
        return session_id

    async def update_status(self, session_id: str, status: str, dag_json: Any = None) -> None:
        if session_id in self._mem_sessions:
            self._mem_sessions[session_id]["status"] = status
        if self._client is not None:
            try:
                patch: dict[str, Any] = {"status": status}
                if dag_json is not None:
                    patch["dag_json"] = dag_json
                self._client.table("simulation_sessions").update(patch).eq(
                    "id", session_id
                ).execute()
            except Exception as exc:  # noqa: BLE001
                logger.warning("update_status failed: %s", exc)

    async def save_report(self, session_id: str, report: dict[str, Any]) -> None:
        self._mem_reports[session_id] = report
        if self._client is not None:
            try:
                for section in report.get("sections", []):
                    self._client.table("report_outputs").insert(
                        {
                            "session_id": session_id,
                            "section": section.get("section"),
                            "content_md": section.get("content"),
                            "chart_data": report.get("chart_data"),
                            "causal_graph": report.get("causal_graph"),
                            "forecast_data": report.get("forecast_data"),
                        }
                    ).execute()
            except Exception as exc:  # noqa: BLE001
                logger.warning("save_report failed: %s", exc)

    async def save_forecast(self, session_id: str, forecast: dict[str, Any]) -> None:
        if self._client is not None:
            try:
                self._client.table("forecast_results").insert(
                    {
                        "session_id": session_id,
                        "model_used": forecast.get("model"),
                        "horizon": forecast.get("horizon_days"),
                        "predictions": forecast.get("predictions"),
                        "mase_score": forecast.get("mase_score"),
                    }
                ).execute()
            except Exception as exc:  # noqa: BLE001
                logger.warning("save_forecast failed: %s", exc)

    async def get_report(self, session_id: str) -> dict[str, Any] | None:
        if session_id in self._mem_reports:
            return self._mem_reports[session_id]
        if self._client is not None:
            try:
                res = (
                    self._client.table("report_outputs")
                    .select("*")
                    .eq("session_id", session_id)
                    .execute()
                )
                if res.data:
                    return {"session_id": session_id, "sections": res.data}
            except Exception as exc:  # noqa: BLE001
                logger.warning("get_report failed: %s", exc)
        return None

    async def list_sessions(self) -> list[dict[str, Any]]:
        if self._client is not None:
            try:
                res = (
                    self._client.table("simulation_sessions")
                    .select("id,query,status,flow_uuid,created_at")
                    .order("created_at", desc=True)
                    .limit(50)
                    .execute()
                )
                if res.data:
                    return res.data
            except Exception as exc:  # noqa: BLE001
                logger.warning("list_sessions failed: %s", exc)
        return list(self._mem_sessions.values())


@lru_cache(maxsize=1)
def get_store() -> Store:
    return Store()
