import { useMemo } from "react";
import type { Data, Layout } from "plotly.js";
import Plot from "@/lib/plotly";
import { OutcomeGaugeSvg } from "@/lib/gauge";
import { useSessionStore } from "@/store/sessionStore";
import { basePlotlyLayout, COLORS, PLOTLY_CONFIG } from "@/lib/theme";

type MetricTone = "data" | "teal" | "orange" | "positive";

interface MetricItem {
  key: string;
  label: string;
  value: string;
  tone?: MetricTone;
}

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

function MetricCell({ label, value, tone = "data" }: MetricItem) {
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

function PendingBlock({ label }: { label: string }) {
  return (
    <div className="flex h-full min-h-0 items-center justify-center rounded-sm border border-dashed border-[color:var(--hairline)] bg-bg/10 font-mono text-[10px] text-muted">
      {label}
    </div>
  );
}

/** Composite prediction overview: outcome gauge, KPI grid, forecast sparkline. */
export function PredictionOverview() {
  const forecast = useSessionStore((s) => s.forecast);
  const causal = useSessionStore((s) => s.causal);
  const deliberation = useSessionStore((s) => s.deliberation);
  const consensus = useSessionStore((s) => s.consensus);
  const rootQuery = useSessionStore((s) => s.rootQuery);

  const overall = causal?.overall_prediction ?? null;
  const hasGauge = overall !== null;
  const hasSpark = Boolean(forecast);
  const queryLine = rootQuery || causal?.root_goal || null;

  const metrics = useMemo((): MetricItem[] => {
    const items: MetricItem[] = [];
    if (consensus) {
      items.push({
        key: "consensus",
        label: "Consensus",
        value: `${(consensus.agreement_score * 100).toFixed(0)}%`,
        tone: "teal",
      });
    }
    if (deliberation) {
      items.push({
        key: "agreement",
        label: "Agreement",
        value: `${(deliberation.agreement_rate * 100).toFixed(0)}%`,
      });
      items.push({
        key: "polarization",
        label: "Polarize",
        value: `${(deliberation.polarization_index * 100).toFixed(0)}%`,
        tone: "orange",
      });
      items.push({
        key: "contagion",
        label: "Contagion",
        value: `${(deliberation.social_contagion_index * 100).toFixed(0)}%`,
      });
      items.push({
        key: "entropy",
        label: "Entropy",
        value: `${(deliberation.entropy_mean * 100).toFixed(0)}%`,
        tone: "teal",
      });
    }
    return items.slice(0, 4);
  }, [consensus, deliberation]);

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

  if (!hasGauge && !hasSpark && metrics.length === 0) {
    return (
      <div className="flex h-full items-center justify-center font-mono text-xs text-muted">
        awaiting prediction synthesis
      </div>
    );
  }

  const metricSlots: (MetricItem | null)[] = [...metrics];
  while (metricSlots.length < 4) metricSlots.push(null);

  return (
    <div className="grid h-full min-h-0 grid-rows-[auto_minmax(0,1fr)_minmax(0,1.1fr)] gap-1 overflow-hidden p-1">
      {/* Query strip — panel header already shows title */}
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

      {/* KPI row — gauge and 2×2 metrics share equal height */}
      <div
        className={`grid min-h-0 items-stretch gap-1 ${
          hasGauge ? "grid-cols-[minmax(5.5rem,40%)_minmax(0,1fr)]" : "grid-cols-1"
        }`}
      >
        {hasGauge && overall !== null ? (
          <OutcomeGauge value={overall} />
        ) : (
          <PendingBlock label="outcome pending" />
        )}

        <div className="grid h-full min-h-0 grid-cols-2 grid-rows-2 gap-0.5 self-stretch">
          {metricSlots.map((m, i) =>
            m ? (
              <MetricCell key={m.key} label={m.label} value={m.value} tone={m.tone} />
            ) : (
              <div key={`empty-${i}`} className="h-full min-h-0 rounded-sm bg-bg/5" aria-hidden />
            )
          )}
        </div>
      </div>

      {/* Trajectory */}
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
