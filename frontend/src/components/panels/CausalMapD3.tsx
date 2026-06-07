import { useEffect, useRef } from "react";
import * as d3 from "d3";
import { useSessionStore } from "@/store/sessionStore";
import { useResize } from "@/hooks/useResize";
import { COLORS } from "@/lib/theme";
import type { CausalNode, CausalNodeKind } from "@/types/events";

type Positioned = CausalNode & { x: number; y: number };

const COLUMN: Record<CausalNodeKind, number> = {
  cause: 0,
  mediator: 1,
  effect: 2,
  goal: 3,
};

const COLUMN_LABEL: Record<number, string> = {
  0: "Cause",
  1: "Mediator",
  2: "Effect",
  3: "Goal",
};

const KIND_COLOR: Record<CausalNodeKind, string> = {
  cause: COLORS.orange,
  mediator: COLORS.teal,
  effect: COLORS.positive,
  goal: COLORS.orange,
};

const KINDS: CausalNodeKind[] = ["cause", "mediator", "effect", "goal"];

function LegendChip({ kind }: { kind: CausalNodeKind }) {
  return (
    <div className="flex items-center justify-center gap-1 font-mono text-[10px]">
      <span
        className="h-1.5 w-1.5 shrink-0 rounded-sm"
        style={{ backgroundColor: KIND_COLOR[kind] }}
      />
      <span className="capitalize text-muted">{kind}</span>
    </div>
  );
}

/**
 * Goal-centric causal flow map with prediction scores, criticality bars,
 * and influence badges on edges.
 */
export function CausalMapD3() {
  const causal = useSessionStore((s) => s.causal);
  const [wrapRef, size] = useResize<HTMLDivElement>();
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();
    if (!causal || size.width === 0 || size.height === 0) return;

    const margin = { top: 18, right: 16, bottom: 12, left: 16 };
    const w = Math.max(0, size.width - margin.left - margin.right);
    const h = Math.max(0, size.height - margin.top - margin.bottom);

    const g = svg
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    const colCount = 4;
    const colPadX = 58;
    const innerW = Math.max(0, w - colPadX * 2);
    const colX = (col: number) =>
      colPadX + (colCount <= 1 ? innerW / 2 : (innerW / (colCount - 1)) * col);

    // Column headers
    for (let col = 0; col < colCount; col += 1) {
      g.append("text")
        .attr("x", colX(col))
        .attr("y", 0)
        .attr("text-anchor", "middle")
        .attr("fill", COLORS.muted)
        .attr("font-family", "'IBM Plex Mono', monospace")
        .attr("font-size", 8)
        .text(COLUMN_LABEL[col]?.toUpperCase() ?? "");
    }

    const graphTop = 10;
    const graphH = Math.max(0, h - graphTop);

    const byColumn = d3.group(causal.nodes, (n) => COLUMN[n.kind]);
    const positioned = new Map<string, Positioned>();

    for (const [col, nodes] of byColumn) {
      const x = colX(col);
      const count = nodes.length;
      const nodeBlockH = 48;
      const gap = Math.max(nodeBlockH, graphH / Math.max(count, 1));
      const totalSpan = gap * (count - 1);
      const startY = graphTop + Math.max(nodeBlockH / 2, (graphH - totalSpan) / 2);

      nodes.forEach((n, i) => {
        positioned.set(n.id, {
          ...n,
          x,
          y: count === 1 ? graphTop + graphH / 2 : startY + gap * i,
        });
      });
    }

    const wScale = d3.scaleLinear().domain([0, 1]).range([1, 8]);

    const linkG = g.append("g");
    causal.edges.forEach((e) => {
      const s = positioned.get(e.source);
      const t = positioned.get(e.target);
      if (!s || !t) return;
      const sig = e.p_value < 0.05;
      const midX = (s.x + t.x) / 2;
      const path = `M${s.x},${s.y} C${midX},${s.y} ${midX},${t.y} ${t.x},${t.y}`;

      linkG
        .append("path")
        .attr("d", path)
        .attr("fill", "none")
        .attr("stroke", sig ? COLORS.teal : COLORS.muted)
        .attr("stroke-opacity", sig ? 0.8 : 0.35)
        .attr("stroke-width", wScale(e.weight))
        .attr("stroke-dasharray", sig ? null : "4 3")
        .attr("stroke-linecap", "round");

      linkG
        .append("text")
        .attr("x", midX)
        .attr("y", (s.y + t.y) / 2 - 5)
        .attr("text-anchor", "middle")
        .attr("fill", sig ? COLORS.teal : COLORS.muted)
        .attr("font-family", "'IBM Plex Mono', monospace")
        .attr("font-size", 8)
        .text(e.influence ?? "+");

      if (Math.abs(t.y - s.y) > 20) {
        linkG
          .append("text")
          .attr("x", midX)
          .attr("y", (s.y + t.y) / 2 + 9)
          .attr("text-anchor", "middle")
          .attr("fill", COLORS.muted)
          .attr("font-family", "'IBM Plex Mono', monospace")
          .attr("font-size", 7)
          .text(`p=${e.p_value.toFixed(2)}`);
      }
    });

    const nodeW = (n: CausalNode) => (n.kind === "goal" ? 108 : 96);
    const nodeH = (n: CausalNode) => (n.kind === "goal" ? 48 : 42);

    const nodeG = g
      .selectAll("g.cnode")
      .data([...positioned.values()])
      .join("g")
      .attr("class", "cnode")
      .attr("transform", (d) => `translate(${d.x},${d.y})`);

    nodeG.each(function (d) {
      const el = d3.select(this);
      const hw = nodeW(d) / 2;
      const hh = nodeH(d) / 2;
      el.append("rect")
        .attr("x", -hw)
        .attr("y", -hh)
        .attr("width", hw * 2)
        .attr("height", hh * 2)
        .attr("rx", 4)
        .attr("fill", COLORS.panelRaised)
        .attr("stroke", KIND_COLOR[d.kind])
        .attr("stroke-width", d.kind === "goal" ? 2 : 1.2);

      el.append("text")
        .attr("text-anchor", "middle")
        .attr("y", -hh + 11)
        .attr("fill", COLORS.data)
        .attr("font-family", "'IBM Plex Mono', monospace")
        .attr("font-size", 8)
        .text(d.label.length > 14 ? `${d.label.slice(0, 13)}…` : d.label);

      el.append("text")
        .attr("text-anchor", "middle")
        .attr("y", -hh + 22)
        .attr("fill", COLORS.orange)
        .attr("font-family", "'IBM Plex Mono', monospace")
        .attr("font-size", 8)
        .text(`${(d.prediction ?? 0).toFixed(0)}%`);

      const barW = hw * 1.5;
      el.append("rect")
        .attr("x", -barW / 2)
        .attr("y", hh - 12)
        .attr("width", barW)
        .attr("height", 3)
        .attr("fill", COLORS.grid)
        .attr("rx", 1.5);
      el.append("rect")
        .attr("x", -barW / 2)
        .attr("y", hh - 12)
        .attr("width", (barW * (d.criticality ?? 0)) / 100)
        .attr("height", 3)
        .attr("fill", COLORS.teal)
        .attr("rx", 1.5);

      el.append("title").text(d.description || d.label);
    });
  }, [causal, size]);

  if (!causal || !causal.nodes.length) {
    return (
      <div className="flex h-full items-center justify-center font-mono text-xs text-muted">
        awaiting causal inference
      </div>
    );
  }

  return (
    <div className="flex h-full min-h-0 flex-col gap-1 overflow-hidden p-1">
      <div className="flex shrink-0 items-start justify-between gap-2">
        <div className="min-w-0 flex-1">
          <span className="font-mono text-[10px] uppercase tracking-wider text-muted">
            Outcome goal
          </span>
          <p
            className="truncate font-mono text-[10px] leading-tight text-data"
            title={causal.root_goal}
          >
            {causal.root_goal || "—"}
          </p>
        </div>
        <div className="shrink-0 rounded-sm border border-[color:var(--hairline)] bg-bg/40 px-2 py-0.5 text-right">
          <span className="block font-mono text-[10px] uppercase text-muted">Overall</span>
          <p className="font-mono text-sm font-semibold leading-none text-orange">
            {(causal.overall_prediction ?? 0).toFixed(1)}%
          </p>
        </div>
      </div>

      <div className="grid shrink-0 grid-cols-4 gap-0.5 border-y border-[color:var(--hairline)] py-0.5">
        {KINDS.map((kind) => (
          <LegendChip key={kind} kind={kind} />
        ))}
      </div>

      <div ref={wrapRef} className="relative min-h-0 flex-1">
        <svg ref={svgRef} width={size.width} height={size.height} className="block h-full w-full" />
      </div>
    </div>
  );
}
