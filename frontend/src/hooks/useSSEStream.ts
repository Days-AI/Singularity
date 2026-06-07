import { useCallback, useEffect, useRef } from "react";
import { getStreamMode, streamUrl, submitQuery } from "@/api/singularity";
import { startMockStream, type MockStreamHandle } from "@/mock/mockStream";
import { SSE_EVENT_TYPES, type SSEvent, type SSEventType } from "@/types/events";
import { useSessionStore } from "@/store/sessionStore";

export const DEFAULT_QUERY =
  "Predict Q4 consumer sentiment for EV market in India";

/**
 * Connects the dashboard to the simulation stream.
 *
 * - In "live" mode it opens an EventSource against /api/stream/{flow_uuid},
 *   subscribing to each named SSE channel. It tracks Last-Event-ID for
 *   resume-after-error semantics (spec section 4.3 error handling).
 * - In "mock" mode it replays the scripted scenario in-browser, requiring no
 *   backend.
 *
 * The returned `start` submits a query and begins streaming; `stop` tears down.
 */
export function useSSEStream() {
  const apply = useSessionStore((s) => s.apply);
  const setConnection = useSessionStore((s) => s.setConnection);
  const reset = useSessionStore((s) => s.reset);

  const esRef = useRef<EventSource | null>(null);
  const mockRef = useRef<MockStreamHandle | null>(null);
  const lastEventId = useRef<string | null>(null);

  const teardown = useCallback(() => {
    esRef.current?.close();
    esRef.current = null;
    mockRef.current?.stop();
    mockRef.current = null;
  }, []);

  const dispatch = useCallback(
    (event: SSEvent) => {
      apply(event);
    },
    [apply]
  );

  const connectLive = useCallback(
    (flowUuid: string) => {
      // Close any prior connection first. Guards against React StrictMode's
      // double-invoked effects racing two async starts into two EventSources.
      teardown();
      const url = streamUrl(flowUuid);
      const es = new EventSource(url, { withCredentials: false });
      esRef.current = es;

      es.onopen = () => setConnection("streaming");
      es.onerror = () => {
        // EventSource auto-reconnects; surface a transient error only if the
        // connection is fully closed.
        if (es.readyState === EventSource.CLOSED) {
          setConnection("error");
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
    [dispatch, setConnection, teardown]
  );

  const start = useCallback(
    async (query: string = DEFAULT_QUERY, questions: string[] = []) => {
      teardown();
      reset();
      setConnection("connecting");
      useSessionStore.getState().setSessionMeta({ rootQuery: query });

      if (getStreamMode() === "mock") {
        mockRef.current = startMockStream(dispatch);
        setConnection("streaming");
        return;
      }

      try {
        const { flowUuid, sessionId } = await submitQuery(query, questions);
        useSessionStore.getState().setSessionMeta({ sessionId });
        connectLive(flowUuid);
      } catch (err) {
        console.error("Failed to start live stream, falling back to mock", err);
        mockRef.current = startMockStream(dispatch);
        setConnection("streaming");
      }
    },
    [connectLive, dispatch, reset, setConnection, teardown]
  );

  const stop = useCallback(() => {
    teardown();
    setConnection("idle");
  }, [setConnection, teardown]);

  useEffect(() => teardown, [teardown]);

  return { start, stop };
}
