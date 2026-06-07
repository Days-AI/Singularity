/** Evenly sample an array for visualization (keeps first/last spread). */
export function downsample<T>(items: T[], max: number): T[] {
  if (items.length <= max) return items;
  const step = items.length / max;
  const out: T[] = [];
  for (let i = 0; i < max; i += 1) {
    out.push(items[Math.min(items.length - 1, Math.floor(i * step))]!);
  }
  return out;
}
