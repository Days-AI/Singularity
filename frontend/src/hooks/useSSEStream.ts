import { useCallback, useEffect, useRef } from "react";
import { streamUrl, submitQuery } from "@/api/singularity";
import { createBatchedDispatch } from "@/lib/sseDispatch";
import { logClient, startClientHeartbeat } from "@/lib/masterLog";
import { SSE_EVENT_TYPES, type SSEvent, type SSEventType } from "@/types/events";
import { useSessionStore } from "@/store/sessionStore";

export const DEFAULT_QUERY =
  "Analyze market sentiment and behavioral drivers for the next quarter";

/**
 * Connects the dashboard to the simulation stream via EventSource against
 * /api/stream/{flow_uuid}, subscribing to each named SSE channel. Tracks
 * Last-Event-ID for resume-after-error semantics (spec section 4.3).
 *
 * The returned `start` submits a query and begins streaming; `stop` tears down.
 */
export function useSSEStream() {
  const apply = useSessionStore((s) => s.apply);
  const setConnection = useSessionStore((s) => s.setConnection);
  const reset = useSessionStore((s) => s.reset);
  const pushToast = useSessionStore((s) => s.pushToast);

  const esRef = useRef<EventSource | null>(null);
  const lastEventId = useRef<string | null>(null);
  const heartbeatStopRef = useRef<(() => void) | null>(null);
  const flowUuidRef = useRef<string | null>(null);

  const stopHeartbeat = useCallback(() => {
    heartbeatStopRef.current?.();
    heartbeatStopRef.current = null;
  }, []);

  const teardown = useCallback(() => {
    esRef.current?.close();
    esRef.current = null;
    stopHeartbeat();
  }, [stopHeartbeat]);

  const dispatchImmediate = useCallback(
    (event: SSEvent) => {
      apply(event);
    },
    [apply]
  );

  const batchedRef = useRef(createBatchedDispatch(dispatchImmediate));
  useEffect(() => {
    batchedRef.current = createBatchedDispatch(dispatchImmediate);
  }, [dispatchImmediate]);

  const dispatch = useCallback(
    (event: SSEvent) => {
      // Lifecycle events must not wait for the next animation frame.
      if (
        event.type === "complete" ||
        event.type === "error" ||
        event.type === "dag_created"
      ) {
        dispatchImmediate(event);
        return;
      }
      batchedRef.current(event);
    },
    [dispatchImmediate]
  );

  const connectLive = useCallback(
    (flowUuid: string) => {
      // Close any prior connection first. Guards against React StrictMode's
      // double-invoked effects racing two async starts into two EventSources.
      teardown();
      const url = streamUrl(flowUuid);
      const es = new EventSource(url, { withCredentials: false });
      esRef.current = es;

      es.onopen = () => {
        setConnection("streaming");
        logClient(
          "stream_connect",
          { flow_uuid: flowUuid },
          { sessionId: useSessionStore.getState().sessionId, flowUuid }
        );
        stopHeartbeat();
        heartbeatStopRef.current = startClientHeartbeat(() => {
          const s = useSessionStore.getState();
          return {
            connection: s.connection,
            sessionId: s.sessionId,
            personasSimulated: s.personasSimulated,
            evidenceCount: s.evidence.length,
            activeAgents: s.activeAgents,
            durationMs: s.durationMs,
            startedAt: s.startedAt,
          };
        });
      };
      es.onerror = () => {
        // EventSource auto-reconnects; surface a transient error only if the
        // connection is fully closed.
        if (es.readyState === EventSource.CLOSED) {
          setConnection("error");
          logClient(
            "stream_error",
            { flow_uuid: flowUuid, ready_state: es.readyState },
            { sessionId: useSessionStore.getState().sessionId, flowUuid }
          );
        }
      };

      const handler = (type: SSEventType) => (ev: MessageEvent) => {
        if (ev.lastEventId) lastEventId.current = ev.lastEventId;
        try {
          const payload = JSON.parse(ev.data);
          dispatch({ type, payload } as SSEvent);
        } catch (err) {
          console.error(`Failed to parse '${type}' SSE payload`, err);
        }
      };

      for (const type of SSE_EVENT_TYPES) {
        es.addEventListener(type, handler(type) as EventListener);
      }

      // The server closes the stream after `complete`. We must close our side
      // synchronously here: otherwise EventSource auto-reconnects when the
      // socket closes and replays the entire scenario, accumulating state.
      // This listener is registered after the dispatch listeners above, so the
      // store has already applied the `complete` event by the time we close.
      es.addEventListener("complete", () => {
        teardown();
      });
    },
    [dispatch, setConnection, stopHeartbeat, teardown]
  );

  const start = useCallback(
    async (
      query: string = DEFAULT_QUERY,
      questions: string[] = [],
      webSourcesEnabled: boolean = true
    ) => {
      teardown();
      reset();
      setConnection("connecting");
      flowUuidRef.current = null;
      useSessionStore.getState().setSessionMeta({ rootQuery: query });

      try {
        const { flowUuid, sessionId } = await submitQuery(
          query,
          questions,
          webSourcesEnabled
        );
        flowUuidRef.current = flowUuid;
        useSessionStore.getState().setSessionMeta({ sessionId });
        logClient(
          "query_submitted",
          { query, questions_count: questions.length },
          { sessionId, flowUuid }
        );
        connectLive(flowUuid);
      } catch (err) {
        console.error("Failed to start live stream", err);
        const msg =
          err instanceof Error ? err.message : "Backend unreachable on :8000";
        logClient("stream_start_failed", { message: msg });
        pushToast({
          kind: "error",
          message: `Live backend unavailable (${msg}). Start run.bat or: cd backend && uvicorn main:app --port 8000.`,
        });
        setConnection("error");
      }
    },
    [connectLive, pushToast, reset, setConnection, teardown]
  );

  const stop = useCallback(() => {
    teardown();
    setConnection("idle");
  }, [setConnection, teardown]);

  useEffect(() => teardown, [teardown]);

  return { start, stop };
}
