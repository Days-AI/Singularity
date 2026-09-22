import { memo } from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";
import { COLORS } from "@/lib/theme";
import type { FlowNodeData } from "@/types/causalIntel";

function MiniSparkline({ values }: { values: number[] }) {
  if (!values.length) return null;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const w = 48;
  const h = 14;
  const pts = values
    .map((v, i) => {
      const x = (i / Math.max(values.length - 1, 1)) * w;
      const y = h - ((v - min) / range) * h;
      return `${x},${y}`;
    })
    .join(" ");
  return (
    <svg width={w} height={h} className="mt-0.5 opacity-80">
      <polyline points={pts} fill="none" stroke={COLORS.teal} strokeWidth="1.2" />
    </svg>
  );
}

function ProbabilityRing({ value }: { value: number }) {
  const pct = Math.max(0, Math.min(100, value));
  const color = pct >= 65 ? COLORS.positive : pct >= 40 ? COLORS.orange : COLORS.alert;
  return (
    <div className="flex items-center gap-1">
      <span className="font-mono text-sm font-bold leading-none" style={{ color }}>
        {pct.toFixed(0)}%
      </span>
      <span className="font-mono text-[9px] uppercase text-muted">prob</span>
    </div>
  );
}

export const CausalCardNode = memo(function CausalCardNode({ data, selected }: NodeProps) {
  const d = data as unknown as FlowNodeData;
  const accent =
    d.nodeType === "risk" ? COLORS.alert : d.nodeType === "market" ? COLORS.positive : COLORS.teal;

  return (
    <div
      className={`w-[168px] rounded-md border bg-panel px-2 py-1.5 shadow-panel transition-shadow ${
        selected ? "ring-1 ring-teal/60" : ""
      }`}
      style={{ borderColor: accent, borderTopWidth: 2 }}
    >
      <Handle type="target" position={Position.Top} className="!h-1.5 !w-1.5 !border-teal !bg-teal" />
      <p className="truncate font-mono text-[10px] font-semibold leading-tight text-data">{d.label}</p>
      <p className="line-clamp-2 font-mono text-[9px] leading-snug text-muted">{d.question}</p>
      <div className="mt-1 flex items-end justify-between gap-1">
        <ProbabilityRing value={d.probability} />
        <span className="font-mono text-[8px] text-muted">{(d.confidence * 100).toFixed(0)}% conf</span>
      </div>
      <div className="mt-0.5 h-1 overflow-hidden rounded-full bg-bg">
        <div
          className="h-full rounded-full bg-teal/70"
          style={{ width: `${Math.min(100, d.criticality)}%` }}
        />
      </div>
      <MiniSparkline values={d.trend} />
      <Handle type="source" position={Position.Bottom} className="!h-1.5 !w-1.5 !border-teal !bg-teal" />
    </div>
  );
});
