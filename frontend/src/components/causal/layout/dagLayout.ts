/** Layered layout helpers for React Flow (goal top, drivers mid, enablers bottom). */
import type { IntelNode } from "@/types/causalIntel";

const LAYER_Y: Record<number, number> = { 0: 40, 1: 220, 2: 400 };

export function layerSort(nodes: IntelNode[]): IntelNode[] {
  return [...nodes].sort((a, b) => a.layer - b.layer || a.label.localeCompare(b.label));
}

export function positionForLayer(layer: number, index: number, count: number): { x: number; y: number } {
  const y = LAYER_Y[layer] ?? 220;
  const x = count <= 1 ? 300 : 80 + (520 / (count - 1)) * index;
  return { x, y };
}

export function groupIds(nodes: IntelNode[]): string[] {
  return [...new Set(nodes.map((n) => n.group_id).filter(Boolean))] as string[];
}
