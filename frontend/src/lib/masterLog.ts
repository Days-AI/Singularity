/**
 * Posts client-side events to the backend master log (logs/master.jsonl).
 * Fire-and-forget; failures are swallowed so logging never breaks the UI.
 */

export interface ClientLogPayload {
  session_id?: string | null;
  flow_uuid?: string | null;
  category?: string;
  event: string;
  phase?: string | null;
  data?: Record<string, unknown>;
}

export function logClient(
  event: string,
  data?: Record<string, unknown>,
  meta?: { sessionId?: string | null; flowUuid?: string | null; category?: string; phase?: string | null }
): void {
  const body: ClientLogPayload = {
    event,
    category: meta?.category ?? "client",
    session_id: meta?.sessionId ?? null,
    flow_uuid: meta?.flowUuid ?? null,
    phase: meta?.phase ?? null,
    data: data ?? {},
  };
  void fetch("/api/log", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).catch(() => undefined);
}

export interface ClientHeartbeatState {
  connection: string;
  sessionId: string | null;
  personasSimulated: number;
  evidenceCount: number;
  activeAgents: number;
  durationMs: number | null;
  startedAt: number | null;
}

const HEARTBEAT_MS = 60_000;

export function startClientHeartbeat(getState: () => ClientHeartbeatState): () => void {
  const tick = () => {
    const s = getState();
    if (s.connection !== "streaming") return;
    const elapsedMs =
      s.durationMs ??
      (s.startedAt != null ? Math.max(0, Date.now() - s.startedAt) : null);
    logClient(
      "heartbeat",
      {
        connection: s.connection,
        personas_simulated: s.personasSimulated,
        evidence_count: s.evidenceCount,
        active_agents: s.activeAgents,
        elapsed_ms: elapsedMs,
      },
      { sessionId: s.sessionId, category: "heartbeat" }
    );
  };
  const id = window.setInterval(tick, HEARTBEAT_MS);
  return () => window.clearInterval(id);
}

export function logSseEvent(
  type: string,
  payload: Record<string, unknown>,
  sessionId: string | null
): void {
  let data: Record<string, unknown> = { type };
  switch (type) {
    case "persona_batch":
      data = {
        batch_index: payload.batch_index,
        profiles_in_batch: payload.profiles_in_batch,
        cumulative_profiles: payload.cumulative_profiles,
      };
      break;
    case "agent_result":
      data = {
        agent_id: payload.agent_id,
        agent_type: payload.agent_type,
        duration_ms: payload.duration_ms,
        confidence: payload.confidence,
        items: Array.isArray(payload.data) ? payload.data.length : 0,
      };
      break;
    case "complete":
      data = {
        duration_ms: payload.duration_ms,
        nodes_resolved: payload.nodes_resolved,
        session_id: payload.session_id,
      };
      break;
    case "error":
      data = {
        code: payload.code,
        message: payload.message,
        node_id: payload.node_id,
      };
      break;
    default:
      return;
  }
  logClient(type, data, { sessionId, category: "sse" });
}
