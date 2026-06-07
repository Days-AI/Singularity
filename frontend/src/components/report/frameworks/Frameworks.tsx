import type { ReportModel } from "@/lib/reportAnalytics";
import { CaptureBlock } from "../charts/CaptureCard";

const TONE: Record<string, string> = {
  positive: "border-positive/40 text-positive",
  alert: "border-alert/40 text-alert",
  neutral: "border-teal/30 text-teal",
};

export function KpiGrid({ model }: { model: ReportModel }) {
  return (
    <CaptureBlock title="Executive KPI Dashboard">
      <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
        {model.kpis.map((k) => (
          <div
            key={k.label}
            className={`rounded-md border bg-panel/70 p-3 ${TONE[k.tone] ?? TONE.neutral}`}
          >
            <div className="font-mono text-2xs uppercase tracking-widest text-muted">{k.label}</div>
            <div className="mt-1 font-display text-xl font-bold leading-none">{k.value}</div>
            <div className="mt-1 font-mono text-2xs text-muted/80">{k.hint}</div>
          </div>
        ))}
      </div>
    </CaptureBlock>
  );
}

function SwotQuadrant({ title, items, accent }: { title: string; items: string[]; accent: string }) {
  return (
    <div className={`rounded-md border bg-panel/60 p-3 ${accent}`}>
      <h5 className="mb-1.5 font-display text-xs font-bold uppercase tracking-wider">{title}</h5>
      <ul className="space-y-1">
        {items.map((it, i) => (
          <li key={i} className="font-mono text-2xs leading-relaxed text-data/90">- {it}</li>
        ))}
      </ul>
    </div>
  );
}

export function SwotMatrix({ model }: { model: ReportModel }) {
  const s = model.swot;
  return (
    <CaptureBlock title="SWOT Analysis">
      <div className="grid grid-cols-1 gap-2 md:grid-cols-2">
        <SwotQuadrant title="Strengths" items={s.strengths} accent="border-positive/40 text-positive" />
        <SwotQuadrant title="Weaknesses" items={s.weaknesses} accent="border-alert/40 text-alert" />
        <SwotQuadrant title="Opportunities" items={s.opportunities} accent="border-teal/40 text-teal" />
        <SwotQuadrant title="Threats" items={s.threats} accent="border-orange/40 text-orange" />
      </div>
    </CaptureBlock>
  );
}

export function PorterForces({ model }: { model: ReportModel }) {
  return (
    <CaptureBlock title="Porter's Five Forces">
      <div className="space-y-2 rounded-md border border-[color:var(--hairline)] bg-panel/60 p-3">
        {model.porter.map((f) => {
          const level = f.intensity > 0.66 ? "High" : f.intensity > 0.4 ? "Moderate" : "Low";
          const color = f.intensity > 0.66 ? "bg-alert" : f.intensity > 0.4 ? "bg-orange" : "bg-positive";
          return (
            <div key={f.force} className="grid grid-cols-[140px_1fr_auto] items-center gap-2">
              <span className="font-mono text-2xs uppercase tracking-wider text-data">{f.force}</span>
              <div className="h-2 overflow-hidden rounded-full bg-bg">
                <div className={`h-full rounded-full ${color}`} style={{ width: `${f.intensity * 100}%` }} />
              </div>
              <span className="w-16 text-right font-mono text-2xs text-muted">{level}</span>
            </div>
          );
        })}
        <div className="pt-1 font-mono text-2xs text-muted/70">
          Intensities are heuristic, derived from simulation signals.
        </div>
      </div>
    </CaptureBlock>
  );
}

export function RecommendationsRoadmap({ model }: { model: ReportModel }) {
  const accents = ["border-positive/40", "border-teal/40", "border-orange/40"];
  return (
    <CaptureBlock title="Strategic Roadmap">
      <div className="grid grid-cols-1 gap-2 md:grid-cols-3">
        {model.recommendations.map((r, i) => (
          <div key={r.horizon} className={`rounded-md border bg-panel/60 p-3 ${accents[i] ?? accents[0]}`}>
            <h5 className="mb-1.5 font-display text-xs font-bold uppercase tracking-wider text-data">{r.horizon}</h5>
            <ul className="space-y-1">
              {r.items.map((it, j) => (
                <li key={j} className="font-mono text-2xs leading-relaxed text-data/90">- {it}</li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </CaptureBlock>
  );
}

export function MarketAssumptions({ model }: { model: ReportModel }) {
  const m = model.marketSizing;
  return (
    <div className="rounded-md border border-orange/30 bg-orange/5 p-3">
      <h5 className="mb-1 font-mono text-2xs uppercase tracking-widest text-orange">
        Market Sizing Assumptions
      </h5>
      <p className="font-mono text-2xs leading-relaxed text-muted">
        Addressable base {m.addressableBase.toLocaleString()} units (normalized). Reach factor{" "}
        {(m.reachFactor * 100).toFixed(0)}% (from positive sentiment); capture rate{" "}
        {(m.adoptionRate * 100).toFixed(0)}% (from mean action-likelihood). TAM ={" "}
        {m.tam.toLocaleString()}, SAM = {m.sam.toLocaleString()}, SOM = {m.som.toLocaleString()}.
        Figures are illustrative and model-derived, not measured market data.
      </p>
    </div>
  );
}
