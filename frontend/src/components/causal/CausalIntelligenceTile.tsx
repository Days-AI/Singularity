import { useCallback, useEffect, useMemo, useState } from "react";
import { ReactFlowProvider } from "@xyflow/react";
import { useCausalIntel } from "@/hooks/useCausalIntel";
import { CausalCanvas } from "./CausalCanvas";
import { NodeInspector } from "./inspector/NodeInspector";
import { ScenarioPanel } from "./scenario/ScenarioPanel";
import {
  HMMStatePanel,
  InfluenceRanking,
  MonteCarloPanel,
  SensitivityPanel,
  TimelineStrip,
} from "./analytics/AnalyticsPanels";
import { fetchAnalytics, fetchHMM, runMonteCarloScenario } from "@/api/causalIntel";
import { groupIds } from "./layout/dagLayout";
import type { AnalyticsBundle, HMMResult } from "@/types/causalIntel";
import { COLORS } from "@/lib/theme";

/** Interactive causal intelligence tile (React Flow canvas + inspector + scenarios). */
export function CausalIntelligenceTile() {
  const {
    graph,
    loading,
    error,
    sessionId,
    scenarioResult,
    assumptions,
    toggleAssumption,
  } = useCausalIntel();

  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [analytics, setAnalytics] = useState<AnalyticsBundle | null>(null);
  const [hmm, setHmm] = useState<HMMResult | null>(null);
  const [showAnalytics, setShowAnalytics] = useState(false);
  const [collapsedGroups, setCollapsedGroups] = useState<Set<string>>(new Set());
  const [mcScenario, setMcScenario] = useState<Record<string, unknown> | null>(null);

  const groups = useMemo(() => (graph ? groupIds(graph.nodes) : []), [graph]);

  useEffect(() => {
    if (!sessionId || !graph) return;
    fetchAnalytics(sessionId).then(setAnalytics).catch(() => setAnalytics(null));
    fetchHMM(sessionId).then(setHmm).catch(() => setHmm(null));
  }, [sessionId, graph]);

  useEffect(() => {
    if (!sessionId || !assumptions.length) return;
    runMonteCarloScenario(sessionId, assumptions)
      .then(setMcScenario)
      .catch(() => setMcScenario(null));
  }, [sessionId, assumptions, scenarioResult]);

  const onNodeClick = useCallback((id: string) => setSelectedNodeId(id), []);

  if (loading && !graph) {
    return (
      <div className="flex h-full items-center justify-center font-mono text-xs text-muted">
        enriching causal graph…
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex h-full items-center justify-center px-2 text-center font-mono text-xs text-alert">
        {error}
      </div>
    );
  }

  if (!graph || !graph.nodes.length) {
    return (
      <div className="flex h-full items-center justify-center font-mono text-xs text-muted">
        awaiting causal inference
      </div>
    );
  }

  const outcome = scenarioResult?.goal_probability ?? graph.overall_prediction;

  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden p-1">
      <header className="flex shrink-0 items-center justify-between gap-2 pb-1">
        <div className="min-w-0 flex-1">
          <span className="font-mono text-[10px] uppercase tracking-wider text-muted">Objective</span>
          <p className="truncate font-mono text-[10px] text-data" title={graph.root_goal}>
            {graph.root_goal}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setShowAnalytics((v) => !v)}
            className="rounded-sm border border-[color:var(--hairline)] px-1.5 py-0.5 font-mono text-[9px] uppercase text-muted hover:border-teal hover:text-teal"
          >
            {showAnalytics ? "Canvas" : "Analytics"}
          </button>
          <div className="rounded-sm border border-[color:var(--hairline)] bg-bg/40 px-2 py-0.5 text-right">
            <span className="block font-mono text-[9px] uppercase text-muted">Outcome</span>
            <p className="font-mono text-sm font-semibold leading-none" style={{ color: COLORS.orange }}>
              {outcome.toFixed(1)}%
            </p>
          </div>
        </div>
      </header>

      {groups.length > 0 && !showAnalytics && (
        <div className="flex shrink-0 flex-wrap gap-1 pb-1">
          {groups.map((g) => {
            const collapsed = collapsedGroups.has(g);
            return (
              <button
                key={g}
                type="button"
                onClick={() =>
                  setCollapsedGroups((prev) => {
                    const next = new Set(prev);
                    if (next.has(g)) next.delete(g);
                    else next.add(g);
                    return next;
                  })
                }
                className={`rounded-sm border px-1.5 py-0.5 font-mono text-[9px] uppercase ${
                  collapsed ? "border-muted/40 text-muted" : "border-teal/40 text-teal"
                }`}
              >
                {collapsed ? "+" : "−"} {g}
              </button>
            );
          })}
        </div>
      )}

      <div className="flex min-h-0 flex-1 overflow-hidden rounded-sm border border-[color:var(--hairline)] bg-bg/10">
        {showAnalytics ? (
          <div className="min-h-0 flex-1 overflow-y-auto p-2 space-y-2">
            {analytics && <InfluenceRanking ranking={analytics.influence_ranking} />}
            {analytics && <SensitivityPanel bars={analytics.sensitivity} />}
            <MonteCarloPanel data={mcScenario ?? graph.monte_carlo ?? analytics?.monte_carlo ?? null} />
            <HMMStatePanel hmm={hmm} />
          </div>
        ) : (
          <>
            <div className="min-h-0 flex-1">
              <ReactFlowProvider>
                <CausalCanvas
                  graph={graph}
                  scenarioResult={scenarioResult}
                  collapsedGroups={collapsedGroups}
                  onNodeClick={onNodeClick}
                />
              </ReactFlowProvider>
            </div>
            <NodeInspector
              nodeId={selectedNodeId}
              sessionId={sessionId}
              onClose={() => setSelectedNodeId(null)}
            />
          </>
        )}
      </div>

      {analytics && <TimelineStrip events={analytics.timeline} />}

      <ScenarioPanel
        graph={graph}
        assumptions={assumptions}
        scenarioResult={scenarioResult}
        onToggle={toggleAssumption}
      />
    </div>
  );
}
