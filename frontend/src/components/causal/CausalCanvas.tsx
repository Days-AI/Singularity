import { useCallback, useMemo } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  type Edge,
  type Node,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import "@/components/causal/causalFlow.css";

import { CausalCardNode } from "./nodes/CausalCardNode";
import { GoalCardNode } from "./nodes/GoalCardNode";
import { PolarityEdge } from "./edges/PolarityEdge";
import type { CausalIntelGraph, ScenarioResult } from "@/types/causalIntel";
import { COLORS } from "@/lib/theme";

const nodeTypes = { causalCard: CausalCardNode, goalCard: GoalCardNode };
const edgeTypes = { polarity: PolarityEdge };

interface CausalCanvasProps {
  graph: CausalIntelGraph;
  scenarioResult: ScenarioResult | null;
  collapsedGroups?: Set<string>;
  onNodeClick: (nodeId: string) => void;
}

export function CausalCanvas({
  graph,
  scenarioResult,
  collapsedGroups,
  onNodeClick,
}: CausalCanvasProps) {
  const hiddenGroups = collapsedGroups ?? new Set<string>();

  const nodes = useMemo((): Node[] => {
    return graph.flow_nodes
      .filter((n) => {
        const gid = n.data.groupId;
        return !gid || !hiddenGroups.has(gid);
      })
      .map((n) => {
      const posterior = scenarioResult?.posterior_by_node?.[n.id];
      const data = { ...n.data };
      if (posterior !== undefined) {
        data.probability = posterior;
      }
      return {
        id: n.id,
        type: n.type,
        position: n.position,
        data,
      };
    });
  }, [graph.flow_nodes, scenarioResult, hiddenGroups]);

  const visibleIds = useMemo(() => new Set(nodes.map((n) => n.id)), [nodes]);

  const edges = useMemo((): Edge[] => {
    return graph.flow_edges
      .filter((e) => visibleIds.has(e.source) && visibleIds.has(e.target))
      .map((e) => ({
      id: e.id,
      source: e.source,
      target: e.target,
      type: e.type,
      data: e.data,
      animated: e.data.polarity === "positive",
    }));
  }, [graph.flow_edges, visibleIds]);

  const onClick = useCallback(
    (_: React.MouseEvent, node: Node) => onNodeClick(node.id),
    [onNodeClick]
  );

  return (
    <div className="h-full min-h-0 w-full">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        onNodeClick={onClick}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        minZoom={0.3}
        maxZoom={1.8}
        proOptions={{ hideAttribution: true }}
        className="causal-flow-root"
      >
        <Background gap={14} size={1} color={COLORS.grid} />
        <Controls showInteractive={false} className="!border-[color:var(--hairline)] !bg-panel/90" />
        <MiniMap
          nodeColor={(n) => (n.type === "goalCard" ? COLORS.teal : COLORS.muted)}
          maskColor="rgba(0,0,0,0.65)"
          className="!border-[color:var(--hairline)] !bg-panel/80"
        />
      </ReactFlow>
    </div>
  );
}
