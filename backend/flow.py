"""SingularityFlow - lightweight async orchestrator.

Mirrors the spec's CrewAI Flow (@start / @listen / @persist, section 3.2) using
plain asyncio so there is no hard CrewAI dependency. Each phase:
  decompose_dag -> evidence (parallel) -> psychometric -> causal -> forecast -> report

Every phase is wrapped so a failure emits an `error` event and the run still
reaches `complete`. State is persisted to the Store at phase boundaries
(the @persist() equivalent).
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from agents import causal as causal_agent
from agents import dag as dag_agent
from agents import evidence as evidence_agent
from agents import forecast as forecast_agent
from agents import psychometric as psychometric_agent
import session_registry
from db.supabase_client import get_store
from report import generate as report_agent
from sse import EventBus
from state import (
    AgentResultPayload,
    AgentStartedPayload,
    CompletePayload,
    DagNode,
    ErrorPayload,
    PersonaBatchPayload,
    SingularityState,
)

logger = logging.getLogger("singularity.flow")


class SingularityFlow:
    def __init__(self, state: SingularityState, bus: EventBus) -> None:
        self.state = state
        self.bus = bus
        self.store = get_store()
        self._started = time.monotonic()
        self._resolved: set[str] = set()

    # --- helpers --------------------------------------------------------------
    async def _emit(self, event_type: str, payload: Any) -> None:
        await self.bus.emit(event_type, payload)

    async def _error(self, code: str, message: str, node_id: str | None = None) -> None:
        logger.warning("flow error [%s] node=%s: %s", code, node_id, message)
        await self._emit("error", ErrorPayload(code=code, message=message, node_id=node_id))

    async def _agent_started(self, node: DagNode) -> None:
        await self._emit(
            "agent_started",
            AgentStartedPayload(agent_id=node.id, task=node.task, agent_type=node.agent_type),
        )

    # --- main entrypoint ------------------------------------------------------
    async def run(self) -> None:
        try:
            await self._decompose_dag()
            await self._run_evidence()
            await self._run_psychometric()
            await self._run_causal()
            await self._run_forecast()
            await self._run_report()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            await self._error("flow_fatal", str(exc))
        finally:
            await self._finish()

    # --- phases ---------------------------------------------------------------
    async def _decompose_dag(self) -> None:
        dag = await dag_agent.decompose(self.state.query)
        self.state.dag = dag
        await self._emit("dag_created", dag)
        await self.store.update_status(
            self.state.session_id or self.state.flow_uuid, "running", dag.model_dump()
        )

    def _nodes_of(self, *agent_types: str) -> list[DagNode]:
        if not self.state.dag:
            return []
        return [n for n in self.state.dag.nodes if n.agent_type in agent_types]

    async def _run_evidence(self) -> None:
        nodes = self._nodes_of("web_search", "financial")
        if not nodes:
            return

        async def run_one(node: DagNode) -> None:
            await self._agent_started(node)
            t0 = time.monotonic()
            try:
                result = await evidence_agent.collect(node, self.state.query)
                self.state.evidence.extend(result.items)
                self.state.series.extend(result.series)
                await self._emit(
                    "agent_result",
                    AgentResultPayload(
                        agent_id=node.id,
                        agent_type=node.agent_type,
                        data=result.items,
                        confidence=result.confidence,
                        duration_ms=int((time.monotonic() - t0) * 1000),
                    ),
                )
                self._resolved.add(node.id)
            except Exception as exc:  # noqa: BLE001
                await self._error("evidence_failed", str(exc), node.id)

        # Bounded parallelism across evidence base_roots.
        await asyncio.gather(*(run_one(n) for n in nodes))

    async def _run_psychometric(self) -> None:
        nodes = self._nodes_of("psychometric")
        node = nodes[0] if nodes else DagNode(
            id="br_psy", task="Psychometric segment simulation", agent_type="psychometric",
        )
        await self._agent_started(node)
        t0 = time.monotonic()

        async def emit_batch(payload: PersonaBatchPayload) -> None:
            await self._emit("persona_batch", payload)

        try:
            result = await psychometric_agent.run(self.state, emit_batch)
            self.state.persona_responses = result.responses
            self.state.ocean_mean = result.ocean_mean
            self.state.metrics["personas"] = result.population
            await self._emit(
                "agent_result",
                AgentResultPayload(
                    agent_id=node.id,
                    agent_type="psychometric",
                    data=result.evidence,
                    confidence=result.confidence,
                    duration_ms=int((time.monotonic() - t0) * 1000),
                ),
            )
            self._resolved.add(node.id)
        except Exception as exc:  # noqa: BLE001
            await self._error("psychometric_failed", str(exc), node.id)

    async def _run_causal(self) -> None:
        try:
            graph = causal_agent.build(self.state)
            self.state.causal = graph
            await self._emit("causal_graph", graph)
        except Exception as exc:  # noqa: BLE001
            await self._error("causal_failed", str(exc))

    async def _run_forecast(self) -> None:
        nodes = self._nodes_of("forecast")
        node = nodes[0] if nodes else DagNode(
            id="br_fct", task="Time-series projection", agent_type="forecast",
        )
        await self._agent_started(node)
        t0 = time.monotonic()
        try:
            forecast = await forecast_agent.run(self.state)
            self.state.forecast = forecast
            await self._emit("forecast_ready", forecast)
            await self._emit(
                "agent_result",
                AgentResultPayload(
                    agent_id=node.id,
                    agent_type="forecast",
                    data=[],
                    confidence=0.85,
                    duration_ms=int((time.monotonic() - t0) * 1000),
                ),
            )
            self._resolved.add(node.id)
            await self.store.save_forecast(
                self.state.session_id or self.state.flow_uuid, forecast.model_dump()
            )
        except Exception as exc:  # noqa: BLE001
            await self._error("forecast_failed", str(exc), node.id)

    async def _run_report(self) -> None:
        try:
            sections = await report_agent.build(self.state)
            self.state.report_sections = sections
            for section in sections:
                await self._emit("report_section", section)
            await self.store.save_report(
                self.state.session_id or self.state.flow_uuid,
                {
                    "sections": [s.model_dump() for s in sections],
                    "causal_graph": self.state.causal.model_dump() if self.state.causal else None,
                    "forecast_data": self.state.forecast.model_dump() if self.state.forecast else None,
                },
            )
        except Exception as exc:  # noqa: BLE001
            await self._error("report_failed", str(exc))

    async def _finish(self) -> None:
        duration_ms = int((time.monotonic() - self._started) * 1000)
        session_id = self.state.session_id or self.state.flow_uuid
        await self.store.update_status(session_id, "complete")
        # Retain the completed state so /api/report/generate can re-synthesize
        # a report on demand without re-running the pipeline.
        session_registry.put(session_id, self.state)
        await self._emit(
            "complete",
            CompletePayload(
                session_id=session_id,
                duration_ms=duration_ms,
                nodes_resolved=len(self._resolved),
            ),
        )
        await self.bus.close()


async def run_persona_preview(stimulus: str, ocean: dict[str, float] | None = None) -> dict[str, Any]:
    """Single-persona preview for POST /api/persona/preview (spec 4.2)."""
    return await psychometric_agent.preview(stimulus=stimulus, ocean=ocean)
