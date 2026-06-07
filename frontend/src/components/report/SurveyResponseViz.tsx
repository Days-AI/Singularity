import { useMemo } from "react";
import { useSessionStore } from "@/store/sessionStore";
import { COLORS } from "@/lib/theme";
import type { PersonaOpinion } from "@/types/events";

/**
 * Survey-style breakdowns derived from the 1,500 persona opinions: a 5-point
 * Likert sentiment scale, behavioral-intent and emotional-state distributions,
 * and an action-likelihood gauge. Rendered as DOM bars so it exports cleanly to
 * PDF/Word via html2canvas.
 */
interface Bar {
  label: string;
  count: number;
  pct: number;
  color: string;
}

const LIKERT = [
  { label: "Strongly negative", min: -1.01, max: -0.5, color: COLORS.alert },
  { label: "Negative", min: -0.5, max: -0.15, color: COLORS.orange },
  { label: "Neutral", min: -0.15, max: 0.15, color: COLORS.muted },
  { label: "Positive", min: 0.15, max: 0.5, color: COLORS.teal },
  { label: "Strongly positive", min: 0.5, max: 1.01, color: COLORS.positive },
];

function tally(values: string[], palette: string[]): Bar[] {
  const counts = new Map<string, number>();
  for (const v of values) {
    if (!v) continue;
    counts.set(v, (counts.get(v) ?? 0) + 1);
  }
  const total = values.length || 1;
  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 6)
    .map(([label, count], i) => ({
      label,
      count,
      pct: (count / total) * 100,
      color: palette[i % palette.length],
    }));
}

function BarGroup({ title, bars }: { title: string; bars: Bar[] }) {
  return (
    <div className="space-y-1.5">
      <h4 className="font-mono text-2xs uppercase tracking-widest text-muted">{title}</h4>
      <div className="space-y-1">
        {bars.map((b) => (
          <div key={b.label} className="flex items-center gap-2">
            <span className="w-32 shrink-0 truncate font-mono text-2xs text-data/80" title={b.label}>
              {b.label}
            </span>
            <div className="relative h-3 flex-1 overflow-hidden rounded-sm bg-bg/60">
              <div
                className="h-full rounded-sm transition-all"
                style={{ width: `${Math.max(b.pct, 1.5)}%`, backgroundColor: b.color, opacity: 0.85 }}
              />
            </div>
            <span className="w-12 shrink-0 text-right font-mono text-2xs tabular-nums text-muted">
              {b.pct.toFixed(0)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

export function SurveyResponseViz({ opinions: override }: { opinions?: PersonaOpinion[] }) {
  const storeOpinions = useSessionStore((s) => s.personaOpinions);
  const opinions = override ?? storeOpinions;

  const { likert, intents, emotions, avgAction } = useMemo(() => {
    const total = opinions.length || 1;
    const likertBars: Bar[] = LIKERT.map((band) => {
      const count = opinions.filter(
        (o) => o.sentiment >= band.min && o.sentiment < band.max
      ).length;
      return { label: band.label, count, pct: (count / total) * 100, color: band.color };
    });
    const intentBars = tally(
      opinions.map((o) => o.behavioral_intent),
      [COLORS.teal, COLORS.positive, COLORS.orange, COLORS.alert, "#9B6DFF", COLORS.muted]
    );
    const emotionBars = tally(
      opinions.map((o) => o.emotional_state),
      [COLORS.orange, COLORS.teal, "#9B6DFF", COLORS.positive, COLORS.alert, COLORS.muted]
    );
    const avg =
      opinions.reduce((acc, o) => acc + o.action_likelihood, 0) / total;
    return { likert: likertBars, intents: intentBars, emotions: emotionBars, avgAction: avg };
  }, [opinions]);

  if (!opinions.length) {
    return (
      <div className="flex h-24 items-center justify-center font-mono text-xs text-muted">
        awaiting persona survey responses
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
      <BarGroup title="Sentiment (5-point Likert)" bars={likert} />
      <div className="space-y-1.5">
        <h4 className="font-mono text-2xs uppercase tracking-widest text-muted">
          Mean action likelihood
        </h4>
        <div className="relative h-6 w-full overflow-hidden rounded-sm bg-bg/60">
          <div
            className="flex h-full items-center justify-end rounded-sm pr-2"
            style={{
              width: `${Math.max(avgAction * 100, 6)}%`,
              backgroundColor: COLORS.teal,
              opacity: 0.85,
            }}
          >
            <span className="font-mono text-2xs font-semibold text-bg">
              {(avgAction * 100).toFixed(0)}%
            </span>
          </div>
        </div>
        <p className="font-mono text-2xs text-muted">
          Share of the {opinions.length.toLocaleString()} simulated personas likely to act.
        </p>
      </div>
      <BarGroup title="Behavioral intent" bars={intents} />
      <BarGroup title="Emotional state" bars={emotions} />
    </div>
  );
}
