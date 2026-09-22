import { memo } from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";
import { COLORS } from "@/lib/theme";
import type { FlowNodeData } from "@/types/causalIntel";

export const GoalCardNode = memo(function GoalCardNode({ data, selected }: NodeProps) {
  const d = data as unknown as FlowNodeData;
  const pct = d.probability;
  const color = pct >= 65 ? COLORS.positive : pct >= 40 ? COLORS.orange : COLORS.alert;

  return (
    <div
      className={`w-[200px] rounded-md border-2 bg-panel px-2.5 py-2 shadow-panel ${
        selected ? "ring-1 ring-teal/60" : ""
      }`}
      style={{ borderColor: COLORS.teal, background: "linear-gradient(180deg, rgba(0,180,216,0.12) 0%, var(--color-panel) 100%)" }}
    >
      <Handle type="target" position={Position.Top} className="!h-2 !w-2 !border-teal !bg-teal" />
      <span className="font-mono text-[9px] uppercase tracking-wider text-teal">Strategic objective</span>
      <p className="mt-0.5 line-clamp-2 font-mono text-[10px] font-semibold leading-tight text-data">
        {d.label}
      </p>
      <p className="mt-1 font-mono text-2xl font-bold leading-none" style={{ color }}>
        {pct.toFixed(1)}%
      </p>
      <span className="font-mono text-[9px] uppercase text-muted">outcome probability</span>
      <Handle type="source" position={Position.Bottom} className="!h-2 !w-2 !border-teal !bg-teal" />
    </div>
  );
});
