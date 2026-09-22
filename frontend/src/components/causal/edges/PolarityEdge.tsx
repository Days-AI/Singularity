import { memo } from "react";
import {
  BaseEdge,
  EdgeLabelRenderer,
  getBezierPath,
  type EdgeProps,
} from "@xyflow/react";
import { COLORS } from "@/lib/theme";

export const PolarityEdge = memo(function PolarityEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  data,
  selected,
}: EdgeProps) {
  const polarity = (data?.polarity as string) ?? "positive";
  const weight = (data?.weight as number) ?? 0.5;
  const label = (data?.label as string) ?? "+";
  const positive = polarity === "positive";
  const color = positive ? COLORS.teal : COLORS.alert;
  const [path, labelX, labelY] = getBezierPath({
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
  });

  return (
    <>
      <BaseEdge
        id={id}
        path={path}
        style={{
          stroke: color,
          strokeWidth: Math.max(1.5, weight * 8),
          strokeOpacity: selected ? 1 : 0.75,
          strokeDasharray: positive ? undefined : "6 4",
        }}
        className={positive ? "causal-edge-flow-positive" : "causal-edge-flow-negative"}
      />
      <EdgeLabelRenderer>
        <div
          className="nodrag nopan pointer-events-none absolute flex h-4 w-4 items-center justify-center rounded-full border font-mono text-[9px] font-bold"
          style={{
            transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)`,
            borderColor: color,
            color,
            background: "var(--color-panel)",
          }}
        >
          {label}
        </div>
      </EdgeLabelRenderer>
    </>
  );
});
