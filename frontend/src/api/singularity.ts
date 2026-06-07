/**
 * Backend API surface. These URLs map onto the FastAPI service (spec section 4.2).
 */

/** SSE stream endpoint for a given flow. Same-origin so Vite can proxy it. */
export function streamUrl(flowUuid: string): string {
  return `/api/stream/${encodeURIComponent(flowUuid)}`;
}

export interface SubmittedQuery {
  flowUuid: string;
  sessionId: string;
}

export interface OllamaHealth {
  reachable: boolean;
  base_url: string;
  configured_model: string;
  active_model?: string;
  model_available: boolean;
  available_models: string[];
  error?: string;
}

export interface BackendHealth {
  status: string;
  ollama_model: string;
  ollama: OllamaHealth;
  openrouter_polish: boolean;
}

/** Probe backend + Ollama readiness (GET /api/health). */
export async function fetchHealth(): Promise<BackendHealth | null> {
  try {
    const res = await fetch("/api/health", { signal: AbortSignal.timeout(4000) });
    if (!res.ok) return null;
    return (await res.json()) as BackendHealth;
  } catch {
    return null;
  }
}

/**
 * Submit a new simulation query. POSTs /api/query and returns flow_uuid +
 * session_id. Throws if the backend is unreachable.
 */
export async function submitQuery(
  query: string,
  questions: string[] = [],
  webSourcesEnabled: boolean = true
): Promise<SubmittedQuery> {
  try {
    const res = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        questions,
        web_sources_enabled: webSourcesEnabled,
      }),
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
    const detail = await res
      .json()
      .then((j) => (j as { detail?: string }).detail)
      .catch(() => undefined);
    throw new Error(detail ?? `query submit failed (${res.status})`);
  } catch (err) {
    throw err instanceof Error
      ? err
      : new Error("Backend unreachable — start it with run.bat or uvicorn on :8000");
  }
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
