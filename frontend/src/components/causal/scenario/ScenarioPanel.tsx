import type { CausalIntelGraph, ScenarioAssumption, ScenarioResult } from "@/types/causalIntel";
import { COLORS } from "@/lib/theme";

interface ScenarioPanelProps {
  graph: CausalIntelGraph;
  assumptions: ScenarioAssumption[];
  scenarioResult: ScenarioResult | null;
  onToggle: (nodeId: string, enabled: boolean) => void;
}

export function ScenarioPanel({
  graph,
  assumptions,
  scenarioResult,
  onToggle,
}: ScenarioPanelProps) {
  const drivers = graph.nodes.filter((n) => n.node_type !== "goal");

  return (
    <div className="shrink-0 border-t border-[color:var(--hairline)] bg-bg/20 px-2 py-1.5">
      <div className="mb-1 flex items-center justify-between">
        <span className="font-mono text-[10px] uppercase tracking-wider text-muted">Scenario simulation</span>
        {scenarioResult && (
          <span className="font-mono text-[10px]" style={{ color: scenarioResult.delta >= 0 ? COLORS.positive : COLORS.alert }}>
            Δ {scenarioResult.delta >= 0 ? "+" : ""}
            {scenarioResult.delta.toFixed(1)}% → {scenarioResult.goal_probability.toFixed(1)}%
          </span>
        )}
      </div>
      <div className="flex flex-wrap gap-1">
        {drivers.map((n) => {
          const a = assumptions.find((x) => x.node_id === n.id);
          const enabled = a?.enabled ?? true;
          return (
            <button
              key={n.id}
              type="button"
              onClick={() => onToggle(n.id, !enabled)}
              className={`rounded-sm border px-1.5 py-0.5 font-mono text-[9px] transition-colors ${
                enabled
                  ? "border-teal/40 bg-teal/10 text-teal"
                  : "border-[color:var(--hairline)] bg-bg/40 text-muted line-through"
              }`}
              title={n.label}
            >
              {n.label.length > 18 ? `${n.label.slice(0, 17)}…` : n.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}
