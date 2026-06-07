export function fmtNum(n: number, digits = 0): string {
  return n.toLocaleString("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

export function fmtPct(n: number, digits = 0): string {
  return `${(n * 100).toFixed(digits)}%`;
}

export function fmtDuration(ms: number | null): string {
  if (ms == null) return "--:--";
  const s = Math.floor(ms / 1000);
  const m = Math.floor(s / 60);
  return `${String(m).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}

export function fmtClock(d: Date): string {
  return d.toLocaleTimeString("en-GB", { hour12: false });
}
