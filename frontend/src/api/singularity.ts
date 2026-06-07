/**
 * Backend API surface. Today these target the mock FastAPI server; the same
 * URLs map onto the real CrewAI-backed FastAPI service later (spec section 4.2)
 * with no frontend changes required.
 */

export type StreamMode = "live" | "mock";

export function getStreamMode(): StreamMode {
  return import.meta.env.VITE_STREAM_MODE === "mock" ? "mock" : "live";
}

/** SSE stream endpoint for a given flow. Same-origin so Vite can proxy it. */
export function streamUrl(flowUuid: string): string {
  return `/api/stream/${encodeURIComponent(flowUuid)}`;
}

export interface SubmittedQuery {
  flowUuid: string;
  sessionId: string;
}

/**
 * Submit a new simulation query. In live mode this POSTs /api/query and the
 * backend returns a flow_uuid + session_id. The mock server also implements
 * this, but to keep the standalone (no-backend) demo working we synthesize a
 * uuid when the request fails.
 */
export async function submitQuery(
  query: string,
  questions: string[] = []
): Promise<SubmittedQuery> {
  try {
    const res = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, questions }),
    });
    if (res.ok) {
      const json = (await res.json()) as {
        flow_uuid: string;
        session_id?: string;
      };
      return {
        flowUuid: json.flow_uuid,
        sessionId: json.session_id ?? json.flow_uuid,
      };
    }
  } catch {
    // fall through to client-side uuid
  }
  const uuid = crypto.randomUUID();
  return { flowUuid: uuid, sessionId: uuid };
}

export interface GeneratedReportSection {
  index: number;
  total: number;
  section: string;
  content: string;
  status: "streaming" | "final";
}

/**
 * Regenerate the strategic report for a completed run, optionally focused by
 * the user's questions. POSTs /api/report/generate and returns the sections.
 */
export async function generateReport(
  sessionId: string,
  questions: string[] = []
): Promise<GeneratedReportSection[]> {
  const res = await fetch("/api/report/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, questions }),
  });
  if (!res.ok) {
    const detail = await res
      .json()
      .then((j) => (j as { detail?: string }).detail)
      .catch(() => undefined);
    throw new Error(detail ?? `report generation failed (${res.status})`);
  }
  const json = (await res.json()) as { sections: GeneratedReportSection[] };
  return json.sections;
}
