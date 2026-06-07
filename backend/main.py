"""Project Singularity backend - FastAPI app + SSE endpoints.

Implements the API surface from spec section 4.2. The real CrewAI-style flow
(flow.py) streams the SSE event contract consumed by the dashboard.

Run: uvicorn main:app --port 8000  (from the backend/ directory)
"""
from __future__ import annotations

from contextlib import asynccontextmanager

import asyncio
import logging
import time
import uuid

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

import session_registry
from config import get_settings
from auth import require_auth
from db.supabase_client import get_store
from flow import SingularityFlow, run_persona_preview
from llm.ollama_client import get_ollama
from observability.master_log import log_entry
from report import generate as report_agent
from sse import EventBus, frame
from state import SingularityState


class QueryRequest(BaseModel):
    query: str
    questions: list[str] = Field(default_factory=list)
    web_sources_enabled: bool = True


class ReportGenerateRequest(BaseModel):
    session_id: str
    questions: list[str] = Field(default_factory=list)


class ClientLogRequest(BaseModel):
    session_id: str | None = None
    flow_uuid: str | None = None
    category: str = "client"
    event: str
    phase: str | None = None
    data: dict = Field(default_factory=dict)


# Simple in-memory rate limit for client log posts: max 120/min per IP.
_client_log_windows: dict[str, tuple[int, float]] = {}
_CLIENT_LOG_LIMIT = 120
_CLIENT_LOG_WINDOW_S = 60.0

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
logger = logging.getLogger("singularity")

settings = get_settings()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Warm-check Ollama on startup so misconfig is visible immediately."""
    ollama = get_ollama()
    st = await ollama.status()
    if not st["reachable"]:
        logger.warning(
            "Ollama unreachable at %s — DAG/persona/report will use fallbacks until it is up.",
            st["base_url"],
        )
    elif not st["model_available"]:
        logger.warning(
            "Ollama is up but model '%s' is missing. Available: %s",
            st["configured_model"],
            ", ".join(st["available_models"][:5]) or "(none)",
        )
    if settings.cognitive_agents_enabled and settings.persona_archetypes > 64:
        logger.warning(
            "PERSONA_ARCHETYPES=%d is high with cognitive agents enabled. "
            "Archetype LLM is skipped when IPIP is loaded; otherwise this adds "
            "significant latency. Recommended: 36.",
            settings.persona_archetypes,
        )
    if settings.cognitive_llm_concurrency > settings.ollama_concurrency:
        logger.warning(
            "COGNITIVE_LLM_CONCURRENCY (%d) exceeds OLLAMA_CONCURRENCY (%d); "
            "effective LLM parallelism is capped at OLLAMA_CONCURRENCY.",
            settings.cognitive_llm_concurrency,
            settings.ollama_concurrency,
        )
    else:
        try:
            await ollama.ensure_model()
            logger.info("Ollama ready: model=%s", ollama.model)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Ollama model check failed: %s", exc)
    yield


app = FastAPI(title="Project Singularity Backend", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins_list,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory registry: flow_uuid -> pending query context. Supabase (if enabled)
# is the durable store; this keeps the live handshake working regardless.
_pending: dict[str, dict] = {}


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


def _check_client_log_rate(ip: str) -> bool:
    now = time.monotonic()
    count, window_start = _client_log_windows.get(ip, (0, now))
    if now - window_start >= _CLIENT_LOG_WINDOW_S:
        count, window_start = 0, now
    count += 1
    _client_log_windows[ip] = (count, window_start)
    return count <= _CLIENT_LOG_LIMIT


@app.get("/api/health")
async def health() -> dict:
    ollama_st = await get_ollama().status()
    return {
        "status": "ok",
        "service": "singularity-backend",
        "version": "0.1.0",
        "ollama_model": settings.ollama_model,
        "ollama": ollama_st,
        "supabase": settings.supabase_enabled,
        "openrouter": settings.openrouter_enabled,
        "openrouter_polish": settings.use_openrouter_polish,
        "personality_engine": not settings.disable_personality_engine,
    }


@app.post("/api/query")
async def submit_query(body: QueryRequest, _auth=Depends(require_auth)) -> JSONResponse:
    query = body.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="query is required")

    questions = [q.strip() for q in body.questions if q.strip()]
    flow_uuid = uuid.uuid4().hex
    store = get_store()
    session_id = await store.create_session(query=query, flow_uuid=flow_uuid)
    _pending[flow_uuid] = {
        "query": query,
        "session_id": session_id,
        "questions": questions,
        "web_sources_enabled": body.web_sources_enabled,
    }
    logger.info("Created flow %s for query: %s", flow_uuid, query)
    log_entry(
        "backend",
        "flow",
        "query_submit",
        session_id=session_id,
        flow_uuid=flow_uuid,
        data={
            "query": query,
            "questions_count": len(questions),
            "web_sources_enabled": body.web_sources_enabled,
        },
    )
    return JSONResponse({"flow_uuid": flow_uuid, "session_id": session_id, "query": query})


@app.get("/api/stream/{flow_uuid}")
async def stream(flow_uuid: str, request: Request, _auth=Depends(require_auth)) -> StreamingResponse:
    ctx = _pending.get(flow_uuid)
    if ctx is None:
        # Allow a stream to start even if /api/query was skipped (resilience):
        # treat the uuid as a fresh ad-hoc run with a default query.
        ctx = {"query": "How does our target audience respond to this campaign message?",
               "session_id": flow_uuid}

    state = SingularityState(
        query=ctx["query"],
        flow_uuid=flow_uuid,
        session_id=ctx.get("session_id"),
        web_sources_enabled=ctx.get("web_sources_enabled", True),
    )
    questions = ctx.get("questions") or []
    if questions:
        state.metrics["focus_questions"] = questions
    bus = EventBus()
    flow = SingularityFlow(state=state, bus=bus)

    async def event_source():
        runner = asyncio.create_task(flow.run())
        event_id = 0
        disconnected = False
        try:
            async for event in bus.drain():
                if await request.is_disconnected():
                    disconnected = True
                    break
                yield frame(event_id, event.type, event.payload)
                event_id += 1
        finally:
            runner.cancel()
            try:
                await runner
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            _pending.pop(flow_uuid, None)
            log_entry(
                "backend",
                "flow",
                "stream_end",
                session_id=state.session_id or flow_uuid,
                flow_uuid=flow_uuid,
                data={"events_sent": event_id, "disconnected": disconnected},
            )

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(event_source(), media_type="text/event-stream", headers=headers)


@app.post("/api/log")
async def client_log(body: ClientLogRequest, request: Request) -> JSONResponse:
    """Append a frontend/client event to the global master log."""
    ip = _client_ip(request)
    if not _check_client_log_rate(ip):
        raise HTTPException(status_code=429, detail="client log rate limit exceeded")
    log_entry(
        "frontend",
        body.category,
        body.event,
        session_id=body.session_id,
        flow_uuid=body.flow_uuid,
        phase=body.phase,
        data=body.data or None,
    )
    return JSONResponse({"ok": True})


@app.post("/api/report/generate")
async def generate_report(body: ReportGenerateRequest, _auth=Depends(require_auth)) -> JSONResponse:
    """Re-synthesize a report for a completed run, optionally focused by the
    user's questions. Reuses the cached SingularityState so the full pipeline
    does not re-run (spec section 3 steps 6-7)."""
    session_id = body.session_id.strip()
    if not session_id:
        raise HTTPException(status_code=422, detail="session_id is required")

    state = session_registry.get(session_id)
    if state is None:
        raise HTTPException(
            status_code=404,
            detail="session not found or expired - run a simulation first",
        )

    questions = [q.strip() for q in body.questions if q.strip()]
    if questions:
        state.metrics["focus_questions"] = questions
    else:
        state.metrics.pop("focus_questions", None)

    sections = await report_agent.build(state)
    state.report_sections = sections
    store = get_store()
    await store.save_report(
        session_id,
        {
            "sections": [s.model_dump() for s in sections],
            "causal_graph": state.causal.model_dump() if state.causal else None,
            "forecast_data": state.forecast.model_dump() if state.forecast else None,
        },
    )
    logger.info("Regenerated report for session %s (questions=%d)", session_id, len(questions))
    return JSONResponse(
        {"session_id": session_id, "sections": [s.model_dump() for s in sections]}
    )


@app.get("/api/report/{session_id}")
async def get_report(session_id: str, _auth=Depends(require_auth)) -> JSONResponse:
    store = get_store()
    report = await store.get_report(session_id)
    if report is None:
        raise HTTPException(status_code=404, detail="report not found")
    return JSONResponse(report)


@app.get("/api/sessions")
async def list_sessions(_auth=Depends(require_auth)) -> JSONResponse:
    store = get_store()
    return JSONResponse({"sessions": await store.list_sessions()})


@app.post("/api/persona/preview")
async def persona_preview(body: dict, _auth=Depends(require_auth)) -> JSONResponse:
    """Simulate a single IPIP-300 persona response to a stimulus (spec 4.2)."""
    stimulus = (body or {}).get("stimulus", "How do you feel about this topic?")
    ocean = (body or {}).get("ocean")  # optional {O,C,E,A,N}
    result = await run_persona_preview(stimulus=stimulus, ocean=ocean)
    return JSONResponse(result)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
