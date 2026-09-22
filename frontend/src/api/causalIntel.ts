import type {
  AnalyticsBundle,
  CausalIntelGraph,
  GraphSnapshot,
  HMMResult,
  NodeDetail,
  PathDiscovery,
  ScenarioAssumption,
  ScenarioResult,
} from "@/types/causalIntel";

async function parseError(res: Response): Promise<string> {
  const j = await res.json().catch(() => ({}));
  return (j as { detail?: string }).detail ?? `request failed (${res.status})`;
}

export async function fetchCausalGraph(
  sessionId: string | null,
  snapshot: GraphSnapshot | null,
  flowUuid: string | null = null
): Promise<CausalIntelGraph> {
  const causal = snapshot?.causal;
  const res = await fetch("/api/causal-intel/graph", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      flow_uuid: flowUuid,
      query: snapshot?.query ?? "",
      causal,
      snapshot: snapshot
        ? {
            query: snapshot.query,
            causal: snapshot.causal,
            evidence: snapshot.evidence,
            forecast: snapshot.forecast,
            metrics: snapshot.metrics,
          }
        : undefined,
    }),
  });
  if (!res.ok) {
    const detail = await parseError(res);
    if (res.status === 404 && detail === "Not Found") {
      throw new Error(
        "Causal intelligence API not loaded — restart the backend (stop.bat then run.bat)."
      );
    }
    throw new Error(detail);
  }
  return (await res.json()) as CausalIntelGraph;
}

export async function fetchNodeDetail(
  nodeId: string,
  sessionId: string | null,
  explain = false
): Promise<NodeDetail> {
  const params = new URLSearchParams();
  if (sessionId) params.set("session_id", sessionId);
  if (explain) params.set("explain", "true");
  const res = await fetch(`/api/causal-intel/nodes/${encodeURIComponent(nodeId)}?${params}`);
  if (!res.ok) throw new Error(await parseError(res));
  return (await res.json()) as NodeDetail;
}

export async function runScenario(
  sessionId: string | null,
  assumptions: ScenarioAssumption[]
): Promise<ScenarioResult> {
  const res = await fetch("/api/causal-intel/scenario", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, assumptions }),
  });
  if (!res.ok) throw new Error(await parseError(res));
  return (await res.json()) as ScenarioResult;
}

export async function fetchAnalytics(sessionId: string): Promise<AnalyticsBundle> {
  const res = await fetch(`/api/causal-intel/analytics/${encodeURIComponent(sessionId)}`);
  if (!res.ok) throw new Error(await parseError(res));
  return (await res.json()) as AnalyticsBundle;
}

export async function fetchHMM(sessionId: string): Promise<HMMResult> {
  const res = await fetch(`/api/causal-intel/hmm/${encodeURIComponent(sessionId)}`);
  if (!res.ok) throw new Error(await parseError(res));
  return (await res.json()) as HMMResult;
}

export async function fetchPaths(
  sessionId: string,
  startId: string,
  endId?: string
): Promise<PathDiscovery> {
  const params = new URLSearchParams({ start_id: startId });
  if (endId) params.set("end_id", endId);
  const res = await fetch(
    `/api/causal-intel/paths/${encodeURIComponent(sessionId)}?${params}`
  );
  if (!res.ok) throw new Error(await parseError(res));
  return (await res.json()) as PathDiscovery;
}

export async function runMonteCarloScenario(
  sessionId: string | null,
  assumptions: ScenarioAssumption[]
): Promise<Record<string, unknown>> {
  const res = await fetch("/api/causal-intel/simulate/monte-carlo", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, assumptions }),
  });
  if (!res.ok) throw new Error(await parseError(res));
  return (await res.json()) as Record<string, unknown>;
}
