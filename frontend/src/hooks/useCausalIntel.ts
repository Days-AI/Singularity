import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useSessionStore } from "@/store/sessionStore";
import { fetchCausalGraph, runScenario } from "@/api/causalIntel";
import type {
  CausalIntelGraph,
  GraphSnapshot,
  ScenarioAssumption,
  ScenarioResult,
} from "@/types/causalIntel";

export function useCausalIntel() {
  const causal = useSessionStore((s) => s.causal);
  const evidence = useSessionStore((s) => s.evidence);
  const forecast = useSessionStore((s) => s.forecast);
  const sessionId = useSessionStore((s) => s.sessionId);
  const flowUuid = useSessionStore((s) => s.flowUuid);
  const rootQuery = useSessionStore((s) => s.rootQuery);
  const deliberation = useSessionStore((s) => s.deliberation);
  const consensus = useSessionStore((s) => s.consensus);

  const [graph, setGraph] = useState<CausalIntelGraph | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [scenarioResult, setScenarioResult] = useState<ScenarioResult | null>(null);
  const [assumptions, setAssumptions] = useState<ScenarioAssumption[]>([]);
  const [intelSessionId, setIntelSessionId] = useState<string | null>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const snapshot = useMemo((): GraphSnapshot | null => {
    if (!causal) return null;
    return {
      query: rootQuery,
      causal,
      evidence: evidence.map(({ source, title, detail, value, unit, url, sentiment }) => ({
        source,
        title,
        detail,
        value,
        unit,
        url,
        sentiment,
      })),
      forecast,
      metrics: {},
      deliberation: deliberation as unknown as Record<string, unknown> | null,
      consensus: consensus as unknown as Record<string, unknown> | null,
    };
  }, [causal, evidence, forecast, rootQuery, deliberation, consensus]);

  useEffect(() => {
    if (!snapshot?.causal?.nodes?.length) {
      setGraph(null);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError(null);
    fetchCausalGraph(sessionId, snapshot, flowUuid)
      .then((g) => {
        if (!cancelled) {
          setGraph(g);
          setIntelSessionId(g.session_id ?? sessionId);
          setAssumptions(
            g.nodes
              .filter((n) => n.node_type !== "goal")
              .map((n) => ({ node_id: n.id, enabled: true }))
          );
        }
      })
      .catch((e) => {
        if (!cancelled) setError(e instanceof Error ? e.message : "failed to load graph");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [snapshot, sessionId, flowUuid]);

  const applyScenario = useCallback(
    (next: ScenarioAssumption[]) => {
      setAssumptions(next);
      const sid = intelSessionId ?? sessionId;
      if (!sid) return;
      if (debounceRef.current) clearTimeout(debounceRef.current);
      debounceRef.current = setTimeout(() => {
        runScenario(sid, next)
          .then(setScenarioResult)
          .catch(() => setScenarioResult(null));
      }, 350);
    },
    [intelSessionId, sessionId]
  );

  const toggleAssumption = useCallback(
    (nodeId: string, enabled: boolean) => {
      const next = assumptions.map((a) =>
        a.node_id === nodeId ? { ...a, enabled } : a
      );
      applyScenario(next);
    },
    [assumptions, applyScenario]
  );

  return {
    graph,
    loading,
    error,
    snapshot,
    sessionId: intelSessionId ?? sessionId,
    scenarioResult,
    assumptions,
    toggleAssumption,
    applyScenario,
  };
}
