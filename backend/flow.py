"""SingularityFlow - lightweight async orchestrator.

Mirrors the spec's CrewAI Flow (@start / @listen / @persist, section 3.2) using
plain asyncio so there is no hard CrewAI dependency. Each phase:
  decompose_dag -> evidence (parallel) -> psychometric ->
  social_simulation -> parallel_post_social (council || analytics) ->
  forecast -> consensus_engine ->
  decision_engine -> report

Every phase is wrapped so a failure emits an `error` event and the run still
reaches `complete`. State is persisted to the Store at phase boundaries
(the @persist() equivalent).
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

from agents import causal as causal_agent
from agents import consensus as consensus_agent
from agents import council as council_agent
from agents import dag as dag_agent
from agents import decision_engine as decision_agent
from agents import evidence as evidence_agent
from agents import forecast as forecast_agent
from agents import monte_carlo as monte_carlo_agent
from agents import prediction_market as prediction_market_agent
from agents import psychometric as psychometric_agent
from agents import social as social_agent
from agents import swarm_optimization as swarm_agent
from config import get_settings
import session_registry
from db.supabase_client import get_store
from observability.master_log import (
    log_algo,
    log_flow,
    log_heartbeat,
    phase_result_summary,
    state_snapshot,
)
from report import generate as report_agent
from sse import EventBus
from state import (
    AgentResultPayload,
    AgentStartedPayload,
    CompletePayload,
    ConsensusPayload,
    CouncilOpinionPayload,
    CouncilReadyPayload,
    DagNode,
    DeliberationPayload,
    ErrorPayload,
    EvidenceItem,
    NarrativeCluster,
    PersonaBatchPayload,
    SingularityState,
    SocialInteractionTickPayload,
    SocialSimulationPayload,
)

logger = logging.getLogger("singularity.flow")


class SingularityFlow:
    def __init__(self, state: SingularityState, bus: EventBus) -> None:
        self.state = state
        self.bus = bus
        self.store = get_store()
        self._started = time.monotonic()
        self._resolved: set[str] = set()
        self._current_phase = "init"
        self._heartbeat_task: asyncio.Task[None] | None = None

    # --- helpers --------------------------------------------------------------
    async def _emit(self, event_type: str, payload: Any) -> None:
        await self.bus.emit(event_type, payload)

    def _elapsed_ms(self) -> int:
        return int((time.monotonic() - self._started) * 1000)

    async def _error(self, code: str, message: str, node_id: str | None = None) -> None:
        logger.warning("flow error [%s] node=%s: %s", code, node_id, message)
        log_entry_data = {"code": code, "message": message, "node_id": node_id}
        log_algo(
            self.state,
            "phase_error",
            self._current_phase,
            elapsed_ms=self._elapsed_ms(),
            data={**state_snapshot(self.state, self._current_phase), **log_entry_data},
        )
        await self._emit("error", ErrorPayload(code=code, message=message, node_id=node_id))

    async def _agent_started(self, node: DagNode) -> None:
        await self._emit(
            "agent_started",
            AgentStartedPayload(agent_id=node.id, task=node.task, agent_type=node.agent_type),
        )

    async def _run_phase(self, name: str, fn: Callable[[], Awaitable[None]]) -> None:
        self._current_phase = name
        log_algo(
            self.state,
            "phase_start",
            name,
            elapsed_ms=self._elapsed_ms(),
            data=state_snapshot(self.state, name),
        )
        t0 = time.monotonic()
        try:
            await fn()
        except Exception:
            raise
        else:
            duration = int((time.monotonic() - t0) * 1000)
            end_data = state_snapshot(self.state, name)
            end_data.update(phase_result_summary(self.state, name))
            log_algo(
                self.state,
                "phase_end",
                name,
                elapsed_ms=self._elapsed_ms(),
                data={**end_data, "phase_duration_ms": duration},
            )

    async def _heartbeat_loop(self) -> None:
        interval = max(1, get_settings().master_log_heartbeat_s)
        try:
            while True:
                await asyncio.sleep(interval)
                log_heartbeat(self.state, self._current_phase, self._elapsed_ms())
        except asyncio.CancelledError:
            raise

    def _start_heartbeat(self) -> None:
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def _stop_heartbeat(self) -> None:
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

    # --- main entrypoint ------------------------------------------------------
    async def run(self) -> None:
        log_flow(
            self.state,
            "flow_start",
            data={
                "query": self.state.query,
                "web_sources_enabled": self.state.web_sources_enabled,
                "focus_questions": self.state.metrics.get("focus_questions", []),
            },
        )
        self._start_heartbeat()
        try:
            await self._run_phase("decompose_dag", self._decompose_dag)
            await self._run_phase("evidence", self._run_evidence)
            await self._run_phase("psychometric", self._run_psychometric)
            await self._run_phase("social_simulation", self._run_social_simulation)
            await self._run_phase("parallel_post_social", self._run_parallel_post_social)
            await self._run_phase("forecast", self._run_forecast)
            await self._run_phase("consensus_engine", self._run_consensus)
            await self._run_phase("decision_engine", self._run_decision_engine)
            await self._run_phase("rag_context", self._prepare_rag_context)
            await self._run_phase("report", self._run_report)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            await self._error("flow_fatal", str(exc))
        finally:
            await self._stop_heartbeat()
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
                result = await evidence_agent.collect(
                    node, self.state.query, self.state.web_sources_enabled
                )
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
            self.state.persona_opinions = result.opinions
            self.state.ocean_mean = result.ocean_mean
            self.state.metrics["personas"] = result.population
            if result.responses:
                import numpy as np

                mean_sent = float(np.mean([r.sentiment_score for r in result.responses]))
                pop = result.population
                self.state.evidence.append(
                    EvidenceItem(
                        source="Simulation",
                        title=f"Population sentiment ({pop:,} personas)",
                        detail=(
                            f"Mean sentiment {mean_sent:+.2f} on -1..1 scale across "
                            f"{len(result.responses)} archetype responses."
                        ),
                        value=round(mean_sent, 3),
                        unit="sentiment",
                        sentiment=mean_sent,
                    )
                )
            if result.deliberation:
                self.state.metrics["deliberation"] = result.deliberation
                clusters = [
                    NarrativeCluster(**c) for c in result.deliberation.get("narrative_clusters", [])
                ]
                await self._emit(
                    "deliberation_ready",
                    DeliberationPayload(
                        agreement_rate=float(result.deliberation.get("agreement_rate", 0.5)),
                        polarization_index=float(result.deliberation.get("polarization_index", 0.2)),
                        confidence_score=float(result.deliberation.get("confidence_score", 0.5)),
                        narrative_clusters=clusters,
                        cluster_sentiments=result.deliberation.get("cluster_sentiments", {}),
                        cluster_actions=result.deliberation.get("cluster_actions", {}),
                        persona_archetypes=result.deliberation.get("persona_archetypes", []),
                        entropy_mean=float(result.deliberation.get("entropy_mean", 0.0)),
                        social_contagion_index=float(
                            result.deliberation.get("social_contagion_index", 0.0)
                        ),
                    ),
                )
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

    async def _run_social_simulation(self) -> None:
        if not get_settings().social_simulation_enabled:
            return
        node = DagNode(
            id="br_soc", task="Social interaction simulation", agent_type="psychometric",
        )
        await self._agent_started(node)
        t0 = time.monotonic()

        async def emit_tick(payload: SocialInteractionTickPayload) -> None:
            await self._emit("social_interaction_tick", payload)

        try:
            result = await social_agent.run(self.state, emit_tick)
            if result.metrics:
                self.state.metrics["social_simulation"] = result.metrics
                refresh = result.metrics.get("deliberation_refresh")
                if refresh:
                    self.state.metrics["deliberation"] = refresh
                    clusters = [
                        NarrativeCluster(**c) for c in refresh.get("narrative_clusters", [])
                    ]
                    await self._emit(
                        "deliberation_ready",
                        DeliberationPayload(
                            agreement_rate=float(refresh.get("agreement_rate", 0.5)),
                            polarization_index=float(refresh.get("polarization_index", 0.2)),
                            confidence_score=float(refresh.get("confidence_score", 0.5)),
                            narrative_clusters=clusters,
                            cluster_sentiments=refresh.get("cluster_sentiments", {}),
                            cluster_actions=refresh.get("cluster_actions", {}),
                            persona_archetypes=refresh.get("persona_archetypes", []),
                            entropy_mean=float(refresh.get("entropy_mean", 0.0)),
                            social_contagion_index=float(refresh.get("social_contagion_index", 0.0)),
                        ),
                    )
            if result.final_payload:
                await self._emit("social_simulation_ready", result.final_payload)
            self._resolved.add(node.id)
            log_algo(
                self.state,
                "phase_end",
                "social_simulation",
                elapsed_ms=int((time.monotonic() - t0) * 1000),
                data={"rounds": result.metrics.get("rounds_completed", 0)},
            )
        except Exception as exc:  # noqa: BLE001
            await self._error("social_simulation_failed", str(exc), node.id)

    async def _run_specialist_council(self) -> None:
        if not get_settings().specialist_council_enabled:
            return
        node = DagNode(
            id="br_council", task="Specialist agent council", agent_type="psychometric",
        )
        await self._agent_started(node)
        t0 = time.monotonic()

        async def emit_opinion(payload: CouncilOpinionPayload) -> None:
            await self._emit("council_opinion", payload)

        try:
            result = await council_agent.run(self.state, emit_opinion)
            self.state.metrics["council"] = result.metrics
            ready = council_agent.to_ready_payload(result)
            await self._emit("council_ready", ready)
            self._resolved.add(node.id)
            log_algo(
                self.state,
                "phase_end",
                "specialist_council",
                elapsed_ms=int((time.monotonic() - t0) * 1000),
                data={"specialists": len(result.opinions)},
            )
        except Exception as exc:  # noqa: BLE001
            await self._error("specialist_council_failed", str(exc), node.id)

    async def _run_parallel_post_social(self) -> None:
        """Council and CPU analytics run concurrently after social simulation."""
        t0 = time.monotonic()
        await asyncio.gather(
            self._run_specialist_council(),
            self._run_analytics_bundle(),
        )
        log_algo(
            self.state,
            "phase_end",
            "parallel_post_social",
            elapsed_ms=int((time.monotonic() - t0) * 1000),
        )

    async def _run_analytics_bundle(self) -> None:
        # Prediction market and Monte Carlo must finish before causal outcome blending.
        await self._run_prediction_market()
        await self._run_monte_carlo()
        await asyncio.gather(
            self._run_causal(),
            self._run_swarm_optimization(),
        )

    async def _refresh_causal_outcome(self) -> None:
        """Recompute headline outcome after forecast (and other late signals) land."""
        if self.state.causal is None:
            return
        overall = round(causal_agent.compute_outcome_probability(self.state), 1)
        if self.state.causal.overall_prediction == overall:
            return
        goal_id = "goal_root"
        nodes = []
        for n in self.state.causal.nodes:
            if n.id == goal_id:
                nodes.append(n.model_copy(update={"prediction": overall}))
            else:
                nodes.append(n)
        self.state.causal = self.state.causal.model_copy(
            update={"overall_prediction": overall, "nodes": nodes}
        )
        await self._emit("causal_graph", self.state.causal)

    async def _run_consensus(self) -> None:
        if not get_settings().consensus_engine_enabled:
            return
        node = DagNode(
            id="br_consensus", task="Consensus engine", agent_type="psychometric",
        )
        await self._agent_started(node)
        t0 = time.monotonic()
        try:
            metrics, payload = consensus_agent.run(self.state)
            self.state.metrics["consensus"] = metrics
            await self._emit("consensus_ready", payload)
            self._resolved.add(node.id)
            log_algo(
                self.state,
                "phase_end",
                "consensus_engine",
                elapsed_ms=int((time.monotonic() - t0) * 1000),
                data={"agreement_score": metrics.get("agreement_score")},
            )
        except Exception as exc:  # noqa: BLE001
            await self._error("consensus_failed", str(exc), node.id)

    async def _run_prediction_market(self) -> None:
        try:
            result = prediction_market_agent.run(self.state)
            self.state.metrics["prediction_market"] = prediction_market_agent.to_metrics(result)
        except Exception as exc:  # noqa: BLE001
            await self._error("prediction_market_failed", str(exc))

    async def _run_monte_carlo(self) -> None:
        try:
            result = monte_carlo_agent.run(self.state)
            self.state.metrics["monte_carlo"] = monte_carlo_agent.to_metrics(result)
        except Exception as exc:  # noqa: BLE001
            await self._error("monte_carlo_failed", str(exc))

    async def _run_causal(self) -> None:
        try:
            graph = causal_agent.build(self.state)
            self.state.causal = graph
            await self._emit("causal_graph", graph)
        except Exception as exc:  # noqa: BLE001
            await self._error("causal_failed", str(exc))

    async def _run_swarm_optimization(self) -> None:
        try:
            result = swarm_agent.run(self.state)
            self.state.metrics["swarm_optimization"] = swarm_agent.to_metrics(result)
        except Exception as exc:  # noqa: BLE001
            await self._error("swarm_optimization_failed", str(exc))

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
            await self._refresh_causal_outcome()
        except Exception as exc:  # noqa: BLE001
            await self._error("forecast_failed", str(exc), node.id)

    async def _run_decision_engine(self) -> None:
        try:
            options = decision_agent.run(self.state)
            self.state.metrics["decision_engine"] = decision_agent.to_metrics(options)
        except Exception as exc:  # noqa: BLE001
            await self._error("decision_engine_failed", str(exc))

    async def _prepare_rag_context(self) -> None:
        if not get_settings().rag_enabled:
            return
        try:
            from rag.retriever import graph_rag_context

            context = await graph_rag_context(self.state, self.state.query)
            if context:
                self.state.metrics["rag_context"] = context
        except Exception as exc:  # noqa: BLE001
            logger.warning("RAG context retrieval skipped: %s", exc)

    async def _index_rag(self) -> None:
        if not get_settings().rag_enabled:
            return
        try:
            from rag.index import index_run

            await index_run(self.state)
        except Exception as exc:  # noqa: BLE001
            logger.warning("RAG indexing skipped: %s", exc)

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
        duration_ms = self._elapsed_ms()
        session_id = self.state.session_id or self.state.flow_uuid
        await self.store.update_status(session_id, "complete")
        session_registry.put(session_id, self.state)
        await self._emit(
            "complete",
            CompletePayload(
                session_id=session_id,
                duration_ms=duration_ms,
                nodes_resolved=len(self._resolved),
            ),
        )
        finish_data = state_snapshot(self.state, self._current_phase)
        finish_data["nodes_resolved"] = len(self._resolved)
        log_flow(
            self.state,
            "flow_complete",
            elapsed_ms=duration_ms,
            data=finish_data,
        )
        if get_settings().rag_enabled:
            asyncio.create_task(self._index_rag_background())
        await self.bus.close()

    async def _index_rag_background(self) -> None:
        try:
            await self._index_rag()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Background RAG indexing failed: %s", exc)


async def run_persona_preview(stimulus: str, ocean: dict[str, float] | None = None) -> dict[str, Any]:
    """Single-persona preview for POST /api/persona/preview (spec 4.2)."""
    return await psychometric_agent.preview(stimulus=stimulus, ocean=ocean)
