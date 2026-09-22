import type { AnalyticsBundle } from "@/types/causalIntel";
import { COLORS } from "@/lib/theme";

export function TimelineStrip({ events }: { events: AnalyticsBundle["timeline"] }) {
  if (!events.length) return null;
  return (
    <div className="shrink-0 border-t border-[color:var(--hairline)] px-2 py-1">
      <span className="font-mono text-[9px] uppercase tracking-wider text-muted">Timeline</span>
      <div className="mt-0.5 flex gap-1 overflow-x-auto pb-0.5">
        {events.slice(0, 12).map((e, i) => (
          <div
            key={`${e.date}-${i}`}
            className="shrink-0 rounded-sm border border-[color:var(--hairline)] bg-bg/30 px-1.5 py-0.5"
            title={e.label}
          >
            <span className="block font-mono text-[8px] uppercase text-muted">{e.kind}</span>
            <span className="block max-w-[72px] truncate font-mono text-[9px] text-data">{e.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export function InfluenceRanking({ ranking }: { ranking: AnalyticsBundle["influence_ranking"] }) {
  if (!ranking.length) return null;
  const max = ranking[0]?.score ?? 1;
  return (
    <div className="space-y-1">
      <span className="font-mono text-[9px] uppercase tracking-wider text-muted">Influence ranking</span>
      {ranking.slice(0, 6).map((r) => (
        <div key={r.node_id} className="flex items-center gap-1">
          <span className="w-16 truncate font-mono text-[9px] text-data">{r.label}</span>
          <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-bg">
            <div
              className="h-full rounded-full bg-teal/70"
              style={{ width: `${(r.score / max) * 100}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

export function SensitivityPanel({ bars }: { bars: AnalyticsBundle["sensitivity"] }) {
  if (!bars.length) return null;
  return (
    <div className="space-y-1">
      <span className="font-mono text-[9px] uppercase tracking-wider text-muted">Sensitivity</span>
      {bars.slice(0, 5).map((b) => (
        <div key={b.node_id} className="font-mono text-[9px]">
          <span className="truncate text-data">{b.label}</span>
          <div className="flex items-center gap-1 text-muted">
            <span>{b.low.toFixed(0)}</span>
            <div className="relative h-1 flex-1 rounded-full bg-bg">
              <div
                className="absolute h-full rounded-full bg-orange/50"
                style={{
                  left: `${b.low}%`,
                  width: `${Math.max(2, b.high - b.low)}%`,
                }}
              />
            </div>
            <span>{b.high.toFixed(0)}</span>
          </div>
        </div>
      ))}
    </div>
  );
}

export function MonteCarloPanel({ data }: { data: Record<string, unknown> | null }) {
  const p = (data?.outcome_percentiles ?? data) as Record<string, number> | undefined;
  if (!p?.p50) return null;
  return (
    <div className="rounded-sm border border-[color:var(--hairline)] bg-bg/30 p-1.5">
      <span className="font-mono text-[9px] uppercase tracking-wider text-muted">Monte Carlo</span>
      <p className="font-mono text-xs text-teal">
        p50 {p.p50}% · p5–p95 {p.p5 ?? "—"}–{p.p95 ?? "—"}
      </p>
    </div>
  );
}

export function HMMStatePanel({ hmm }: { hmm: { current_state: string; states: { name: string; probability: number }[] } | null }) {
  if (!hmm) return null;
  return (
    <div className="rounded-sm border border-[color:var(--hairline)] bg-bg/30 p-1.5">
      <span className="font-mono text-[9px] uppercase tracking-wider text-muted">HMM state</span>
      <p className="font-mono text-xs" style={{ color: COLORS.teal }}>
        Current: {hmm.current_state}
      </p>
      <div className="mt-0.5 flex flex-wrap gap-1">
        {hmm.states.map((s) => (
          <span key={s.name} className="font-mono text-[8px] text-muted">
            {s.name} {(s.probability * 100).toFixed(0)}%
          </span>
        ))}
      </div>
    </div>
  );
}
