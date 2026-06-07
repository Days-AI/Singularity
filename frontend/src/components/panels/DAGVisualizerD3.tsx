import { useEffect, useRef } from "react";
import * as d3 from "d3";
import { useSessionStore } from "@/store/sessionStore";
import { useResize } from "@/hooks/useResize";
import { COLORS } from "@/lib/theme";
import type { DagNode, NodeStatus } from "@/types/events";

interface SimNode extends d3.SimulationNodeDatum, DagNode {}
interface SimLink extends d3.SimulationLinkDatum<SimNode> {
  source: string | SimNode;
  target: string | SimNode;
}

const STATUS_FILL: Record<NodeStatus, string> = {
  pending: COLORS.muted,
  running: COLORS.orange,
  done: COLORS.positive,
  failed: COLORS.alert,
};

const AGENT_GLYPH: Record<string, string> = {
  web_search: "WEB",
  financial: "FIN",
  psychometric: "PSY",
  forecast: "FCT",
};

/**
 * Force-directed DAG. Nodes are created once when dag_created arrives; on every
 * status change we recolor in place without restarting the layout, so the
 * graph settles once and then animates state transitions smoothly.
 */
export function DAGVisualizerD3() {
  const dagNodes = useSessionStore((s) => s.dagNodes);
  const dagEdges = useSessionStore((s) => s.dagEdges);
  const nodeStatus = useSessionStore((s) => s.nodeStatus);

  const [wrapRef, size] = useResize<HTMLDivElement>();
  const svgRef = useRef<SVGSVGElement>(null);
  const simRef = useRef<d3.Simulation<SimNode, SimLink> | null>(null);

  // (Re)build graph when topology or canvas size changes.
  useEffect(() => {
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();
    if (!dagNodes.length || size.width === 0 || size.height === 0) return;

    const { width, height } = size;
    const nodes: SimNode[] = dagNodes.map((n) => ({ ...n }));
    const links: SimLink[] = dagEdges.map((e) => ({
      source: e.source,
      target: e.target,
    }));

    const defs = svg.append("defs");
    defs
      .append("marker")
      .attr("id", "dag-arrow")
      .attr("viewBox", "0 -5 10 10")
      .attr("refX", 22)
      .attr("refY", 0)
      .attr("markerWidth", 6)
      .attr("markerHeight", 6)
      .attr("orient", "auto")
      .append("path")
      .attr("d", "M0,-5L10,0L0,5")
      .attr("fill", COLORS.muted);

    const root = svg.append("g");

    const link = root
      .append("g")
      .attr("stroke", COLORS.grid)
      .attr("stroke-width", 1.2)
      .selectAll("line")
      .data(links)
      .join("line")
      .attr("marker-end", "url(#dag-arrow)");

    const node = root
      .append("g")
      .selectAll<SVGGElement, SimNode>("g")
      .data(nodes, (d) => d.id)
      .join("g")
      .attr("class", "dag-node")
      .style("cursor", "grab");

    node
      .append("circle")
      .attr("r", 14)
      .attr("fill", COLORS.panelRaised)
      .attr("stroke", (d) => STATUS_FILL[nodeStatus[d.id] ?? "pending"])
      .attr("stroke-width", 2);

    node
      .append("circle")
      .attr("class", "status-core")
      .attr("r", 5)
      .attr("fill", (d) => STATUS_FILL[nodeStatus[d.id] ?? "pending"]);

    node
      .append("text")
      .attr("dy", -20)
      .attr("text-anchor", "middle")
      .attr("fill", COLORS.muted)
      .attr("font-family", "'IBM Plex Mono', monospace")
      .attr("font-size", 8)
      .text((d) => AGENT_GLYPH[d.agent_type] ?? d.agent_type);

    node.append("title").text((d) => `${d.id}: ${d.task}`);

    const sim = d3
      .forceSimulation<SimNode>(nodes)
      .force(
        "link",
        d3
          .forceLink<SimNode, SimLink>(links)
          .id((d) => d.id)
          .distance(90)
          .strength(0.6)
      )
      .force("charge", d3.forceManyBody().strength(-340))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collide", d3.forceCollide(28))
      .force("x", d3.forceX(width / 2).strength(0.05))
      .force("y", d3.forceY(height / 2).strength(0.08));

    simRef.current = sim;

    sim.on("tick", () => {
      nodes.forEach((n) => {
        n.x = Math.max(20, Math.min(width - 20, n.x ?? width / 2));
        n.y = Math.max(20, Math.min(height - 20, n.y ?? height / 2));
      });
      link
        .attr("x1", (d) => (d.source as SimNode).x ?? 0)
        .attr("y1", (d) => (d.source as SimNode).y ?? 0)
        .attr("x2", (d) => (d.target as SimNode).x ?? 0)
        .attr("y2", (d) => (d.target as SimNode).y ?? 0);
      node.attr("transform", (d) => `translate(${d.x},${d.y})`);
    });

    const drag = d3
      .drag<SVGGElement, SimNode>()
      .on("start", (event, d) => {
        if (!event.active) sim.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
      })
      .on("drag", (event, d) => {
        d.fx = event.x;
        d.fy = event.y;
      })
      .on("end", (event, d) => {
        if (!event.active) sim.alphaTarget(0);
        d.fx = null;
        d.fy = null;
      });
    node.call(drag);

    return () => {
      sim.stop();
    };
  }, [dagNodes, dagEdges, size]);

  // Recolor on status change without restarting layout. Running nodes pulse
  // via a CSS animation toggled with the `is-running` class (see globals.css),
  // which avoids self-perpetuating D3 transitions.
  useEffect(() => {
    const svg = d3.select(svgRef.current);
    svg.selectAll<SVGGElement, SimNode>("g.dag-node").each(function (d) {
      const status = nodeStatus[d.id] ?? "pending";
      const fill = STATUS_FILL[status];
      const g = d3.select(this);
      g.select<SVGCircleElement>("circle").transition().duration(400).attr("stroke", fill);
      g.select<SVGCircleElement>("circle.status-core")
        .transition()
        .duration(400)
        .attr("fill", fill);
      g.classed("is-running", status === "running");
    });
  }, [nodeStatus]);

  return (
    <div ref={wrapRef} className="h-full w-full">
      {dagNodes.length === 0 ? (
        <EmptyHint />
      ) : (
        <svg ref={svgRef} width={size.width} height={size.height} className="block" />
      )}
    </div>
  );
}

function EmptyHint() {
  return (
    <div className="flex h-full items-center justify-center text-center font-mono text-xs text-muted">
      <span>DAG decomposition pending //<br />run a simulation to populate</span>
    </div>
  );
}
