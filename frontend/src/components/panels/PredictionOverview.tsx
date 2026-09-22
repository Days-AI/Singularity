import { useMemo } from "react";
import type { Data, Layout } from "plotly.js";
import Plot from "@/lib/plotly";
import { OutcomeGaugeSvg } from "@/lib/gauge";
import {
  buildPredictionOverviewModel,
  type PredictionOverviewKpi,
} from "@/lib/reportAnalytics";
import { useSessionStore } from "@/store/sessionStore";
import { basePlotlyLayout, COLORS, PLOTLY_CONFIG } from "@/lib/theme";

type MetricTone = "data" | "teal" | "orange" | "positive";

const TONE_CLASS: Record<MetricTone, string> = {
  data: "text-data",
  teal: "text-teal",
  orange: "text-orange",
  positive: "text-[color:var(--color-positive)]",
};

const TONE_ACCENT: Record<MetricTone, string> = {
  data: COLORS.data,
  teal: COLORS.teal,
  orange: COLORS.orange,
  positive: COLORS.positive,
};

function OutcomeGauge({ value }: { value: number }) {
  return (
    <div className="grid h-full min-h-0 w-full place-items-center rounded-sm border border-[color:var(--hairline)] bg-bg/30 p-0.5">
      <div className="aspect-[5/3] h-full max-h-full w-full max-w-[9.5rem]">
        <OutcomeGaugeSvg value={value} label="OUTCOME" className="h-full w-full" />
      </div>
    </div>
  );
}

function MetricCell({ label, value, tone = "data" }: PredictionOverviewKpi) {
  const accent = TONE_ACCENT[tone];
  return (
    <div
      className="flex h-full min-h-0 flex-col justify-center rounded-sm border border-[color:var(--hairline)] bg-bg/40 px-1.5 py-0.5"
      style={{ borderTopColor: accent, borderTopWidth: 2 }}
    >
      <span className="truncate font-mono text-[10px] uppercase tracking-wide text-muted">
        {label}
      </span>
      <p className={`font-mono text-xs font-semibold leading-tight ${TONE_CLASS[tone]}`}>
        {value}
      </p>
    </div>
  );
}

function StreamBreakdown({
  bars,
}: {
  bars: { key: string; label: string; value: number; color: string }[];
}) {
  if (!bars.length) return null;
  return (
    <div className="flex min-w-0 shrink-0 flex-col gap-0.5">
      {bars.map((b) => (
        <div key={b.key} className="flex items-center gap-1 font-mono text-[9px]">
          <span className="w-[4.5rem] shrink-0 truncate uppercase text-muted">{b.label}</span>
          <div className="h-1 min-w-0 flex-1 overflow-hidden rounded-sm bg-bg/50">
            <div
              className="h-full rounded-sm"
              style={{ width: `${Math.min(100, b.value)}%`, backgroundColor: b.color }}
            />
          </div>
          <span className="w-7 shrink-0 text-right text-data">{b.value.toFixed(0)}</span>
        </div>
      ))}
    </div>
  );
}

function PendingBlock({ label }: { label: string }) {
  return (
    <div className="flex h-full min-h-0 items-center justify-center rounded-sm border border-dashed border-[color:var(--hairline)] bg-bg/10 font-mono text-[10px] text-muted">
      {label}
    </div>
  );
}

/** Composite prediction overview: outcome gauge, stream breakdown, KPI grid, forecast sparkline. */
export function PredictionOverview() {
  const forecast = useSessionStore((s) => s.forecast);
  const causal = useSessionStore((s) => s.causal);
  const deliberation = useSessionStore((s) => s.deliberation);
  const consensus = useSessionStore((s) => s.consensus);
  const predictionMarket = useSessionStore((s) => s.predictionMarket);
  const monteCarlo = useSessionStore((s) => s.monteCarlo);
  const personaOpinions = useSessionStore((s) => s.personaOpinions);
  const rootQuery = useSessionStore((s) => s.rootQuery);

  const model = useMemo(
    () =>
      buildPredictionOverviewModel({
        causal,
        deliberation,
        consensus,
        predictionMarket,
        monteCarlo,
        personaOpinions,
      }),
    [causal, deliberation, consensus, predictionMarket, monteCarlo, personaOpinions]
  );

  const hasGauge = model.overall !== null;
  const hasSpark = Boolean(forecast);
  const queryLine = rootQuery || causal?.root_goal || null;

  const { sparkData, sparkLayout } = useMemo(() => {
    if (!forecast) return { sparkData: [] as Data[], sparkLayout: {} as Partial<Layout> };

    const history = forecast.history.slice(-24);
    const histXs = history.map((p) => p.date);
    const histYs = history.map((p) => p.value);
    const predXs = forecast.predictions.map((p) => p.date);
    const predYs = forecast.predictions.map((p) => p.value);
    const pivot = histXs.length > 0 ? histXs[histXs.length - 1] : predXs[0];

    const data: Data[] = [
      {
        x: histXs,
        y: histYs,
        type: "scatter",
        mode: "lines",
        name: "History",
        line: { color: COLORS.muted, width: 1.4 },
        hovertemplate: "%{x}<br>%{y:.1f}<extra>history</extra>",
      },
      {
        x: predXs,
        y: predYs,
        type: "scatter",
        mode: "lines",
        name: "Forecast",
        line: { color: COLORS.teal, width: 2, dash: "dot" },
        fill: "tozeroy",
        fillcolor: "rgba(0,180,216,0.12)",
        hovertemplate: "%{x}<br>%{y:.1f}<extra>forecast</extra>",
      },
    ];

    const layout: Partial<Layout> = {
      shapes: pivot
        ? [
            {
              type: "line",
              x0: pivot,
              x1: pivot,
              y0: 0,
              y1: 1,
              yref: "paper",
              line: { color: COLORS.orange, width: 1, dash: "dot" },
            },
          ]
        : [],
    };

    return { sparkData: data, sparkLayout: layout };
  }, [forecast]);

  if (!hasGauge && !hasSpark && model.kpis.length === 0) {
    return (
      <div className="flex h-full items-center justify-center font-mono text-xs text-muted">
        awaiting prediction synthesis
      </div>
    );
  }

  const metricSlots: (PredictionOverviewKpi | null)[] = [...model.kpis];
  while (metricSlots.length < 6) metricSlots.push(null);

  return (
    <div className="grid h-full min-h-0 grid-rows-[auto_auto_minmax(0,1fr)_minmax(0,1.1fr)] gap-1 overflow-hidden p-1">
      <div className="flex min-w-0 shrink-0 items-center justify-between gap-2 font-mono text-[10px]">
        <p
          className="min-w-0 flex-1 truncate uppercase tracking-wider text-muted"
          title={queryLine ?? undefined}
        >
          {queryLine ?? "Awaiting query"}
        </p>
        {consensus && (
          <span className="shrink-0 rounded-sm border border-teal/30 bg-teal/10 px-1 py-px text-teal">
            Σ {(consensus.agreement_score * 100).toFixed(0)}%
          </span>
        )}
      </div>

      <StreamBreakdown bars={model.streamBars} />

      <div
        className={`grid min-h-0 items-stretch gap-1 ${
          hasGauge ? "grid-cols-[minmax(5.5rem,38%)_minmax(0,1fr)]" : "grid-cols-1"
        }`}
      >
        {hasGauge && model.overall !== null ? (
          <div className="flex min-h-0 flex-col gap-0.5">
            <OutcomeGauge value={model.overall} />
            {model.marketCi && (
              <p className="text-center font-mono text-[9px] text-muted">
                mkt CI {model.marketCi}
              </p>
            )}
          </div>
        ) : (
          <PendingBlock label="outcome pending" />
        )}

        <div className="grid h-full min-h-0 grid-cols-2 grid-rows-3 gap-0.5 self-stretch">
          {metricSlots.map((m, i) =>
            m ? (
              <MetricCell key={m.key} label={m.label} value={m.value} tone={m.tone} />
            ) : (
              <div key={`empty-${i}`} className="h-full min-h-0 rounded-sm bg-bg/5" aria-hidden />
            )
          )}
        </div>
      </div>

      <div className="flex min-h-0 flex-col overflow-hidden rounded-sm border border-[color:var(--hairline)] bg-bg/15">
        <div className="flex shrink-0 items-center justify-between gap-2 px-1.5 py-0.5 font-mono text-[10px]">
          <span className="uppercase tracking-wider text-muted">Trajectory</span>
          {forecast && (
            <span className="truncate text-teal/80" title={forecast.model}>
              {forecast.model}
            </span>
          )}
        </div>
        <div className="relative min-h-0 flex-1">
          {hasSpark ? (
            <Plot
              data={sparkData}
              useResizeHandler
              style={{ width: "100%", height: "100%" }}
              layout={{
                ...basePlotlyLayout(),
                ...sparkLayout,
                margin: { l: 30, r: 4, t: 4, b: 20 },
                showlegend: false,
                xaxis: {
                  gridcolor: COLORS.grid,
                  tickfont: { size: 7, color: COLORS.muted },
                  nticks: 4,
                  showgrid: true,
                  zeroline: false,
                },
                yaxis: {
                  gridcolor: COLORS.grid,
                  tickfont: { size: 7, color: COLORS.muted },
                  nticks: 4,
                  showgrid: true,
                  zeroline: false,
                },
              }}
              config={PLOTLY_CONFIG}
            />
          ) : (
            <PendingBlock label="forecast pending" />
          )}
        </div>
      </div>
    </div>
  );
}
