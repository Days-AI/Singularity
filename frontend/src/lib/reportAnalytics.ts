/**
 * Report analytics layer.
 *
 * Pure, deterministic derivations that turn the live simulation state (persona
 * opinions, OCEAN aggregates, evidence, forecast, causal graph) into the
 * consulting-grade metrics and framework inputs the enterprise report renders:
 * KPIs, demand/market sizing, scenarios, sensitivity, SWOT, Porter Five Forces,
 * and the growth / risk matrices.
 *
 * IMPORTANT - provenance: this product has no external market-data feed. Any
 * absolute market size (TAM/SAM/SOM) or ROI figure here is an ASSUMPTION-DRIVEN,
 * MODEL-DERIVED estimate computed from the simulated population and sentiment,
 * not a measured market value. The report surfaces this via `DISCLAIMER` and an
 * explicit assumptions block so the numbers are never mistaken for ground truth.
 */
import type {
  ConsensusPayload,
  CouncilOpinionPayload,
  CouncilReadyPayload,
  CausalGraphPayload,
  DeliberationPayload,
  ForecastReadyPayload,
  HeatmapRow,
  OceanScores,
  PersonaOpinion,
  PersonaPoint,
  SocialSimulationPayload,
} from "@/types/events";
import type { EvidenceEntry, ReportSection } from "@/store/sessionStore";
import { parseApplicationSection, type ParsedPlaybook } from "@/lib/useCases";

export const DISCLAIMER =
  "Market sizing (TAM/SAM/SOM) and ROI are illustrative, assumption-driven " +
  "estimates derived from the simulated persona population and sentiment - not " +
  "measured market data. Treat them as directional, not authoritative.";

export interface Kpi {
  label: string;
  value: string;
  hint: string;
  tone: "positive" | "neutral" | "alert";
}

export interface ClusterStat {
  label: string;
  size: number;
  share: number; // 0..1
  meanSentiment: number; // -1..1
  meanAction: number; // 0..1
}

export interface MarketSizing {
  tam: number;
  sam: number;
  som: number;
  addressableBase: number;
  reachFactor: number;
  adoptionRate: number;
}

export interface ScenarioStat {
  name: "Pessimistic" | "Base" | "Optimistic";
  endValue: number;
  pctChange: number;
}

export interface NamedWeight {
  label: string;
  weight: number;
}

export interface MatrixPoint {
  label: string;
  x: number;
  y: number;
  size: number;
}

export interface Swot {
  strengths: string[];
  weaknesses: string[];
  opportunities: string[];
  threats: string[];
}

export interface PorterForce {
  force: string;
  intensity: number; // 0..1
  note: string;
}

export interface EvidenceFinding {
  source: string;
  title: string;
  detail: string;
  url?: string;
  sentiment?: number;
}

export interface SourceBreakdownRow {
  source: string;
  count: number;
  meanSentiment: number;
  sampleTitles: string[];
}

export interface Recommendation {
  horizon: "Quick Wins (0-3mo)" | "Medium-Term (3-12mo)" | "Long-Term (12mo+)";
  items: string[];
}

export interface MetricBar {
  label: string;
  value: number;
}

export interface SocialNarrativeRow {
  label: string;
  adoptionPct: number;
  sentiment: number;
}

export interface CouncilConfidenceRow {
  role: string;
  confidence: number;
}

export interface ConsensusScores {
  agreement: number;
  councilAlignment: number;
  recommendedAction: string;
}

export interface ReportModel {
  query: string;
  generatedAt: string;
  population: number;
  sampleResponses: number;
  meanSentiment: number;
  positiveShare: number;
  negativeShare: number;
  neutralShare: number;
  adoptionRate: number;
  oceanMean: OceanScores | null;
  forecastGrowthPct: number | null;
  forecastModel: string | null;
  forecastMase: number | null;
  kpis: Kpi[];
  clusters: ClusterStat[];
  marketSizing: MarketSizing;
  scenarios: ScenarioStat[];
  sensitivity: NamedWeight[];
  growthMatrix: MatrixPoint[];
  riskMatrix: MatrixPoint[];
  swot: Swot;
  porter: PorterForce[];
  topConcerns: NamedWeight[];
  evidenceBySource: SourceBreakdownRow[];
  evidenceFindings: EvidenceFinding[];
  externalIntelligenceContent: string | null;
  councilConsensusContent: string | null;
  causalOutcome: number | null;
  deliberationMetrics: MetricBar[] | null;
  consensusScores: ConsensusScores | null;
  socialNarratives: SocialNarrativeRow[];
  councilConfidence: CouncilConfidenceRow[];
  recommendations: Recommendation[];
  narrative: { title: string; content: string }[];
  applicationPlaybooks: ParsedPlaybook[];
  heatmap: HeatmapRow[];
  personaPoints: PersonaPoint[];
  causal: CausalGraphPayload | null;
  forecast: ForecastReadyPayload | null;
}

export interface AnalyticsInput {
  query: string;
  evidence: EvidenceEntry[];
  oceanMean: OceanScores | null;
  personaPoints: PersonaPoint[];
  personaOpinions: PersonaOpinion[];
  heatmap: HeatmapRow[];
  personasSimulated: number;
  forecast: ForecastReadyPayload | null;
  causal: CausalGraphPayload | null;
  reportSections: ReportSection[];
  deliberation: DeliberationPayload | null;
  consensus: ConsensusPayload | null;
  socialSimulation: SocialSimulationPayload | null;
  council: CouncilReadyPayload | null;
  councilOpinions: CouncilOpinionPayload[];
}

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));
const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0);
const pct = (v: number) => `${(v * 100).toFixed(0)}%`;

function tone(sentiment: number): "positive" | "neutral" | "alert" {
  if (sentiment > 0.15) return "positive";
  if (sentiment < -0.15) return "alert";
  return "neutral";
}

function deriveClusters(opinions: PersonaOpinion[], population: number): ClusterStat[] {
  if (!opinions.length) return [];
  const groups = new Map<string, PersonaOpinion[]>();
  for (const o of opinions) {
    const key = o.cluster_label || `Cluster ${o.cluster}`;
    const arr = groups.get(key) ?? [];
    arr.push(o);
    groups.set(key, arr);
  }
  const sampleTotal = opinions.length;
  return [...groups.entries()]
    .map(([label, arr]) => {
      const share = arr.length / sampleTotal;
      return {
        label,
        size: Math.round(share * population),
        share,
        meanSentiment: mean(arr.map((o) => o.sentiment)),
        meanAction: mean(arr.map((o) => o.action_likelihood)),
      };
    })
    .sort((a, b) => b.size - a.size);
}

function deriveMarketSizing(positiveShare: number, adoptionRate: number): MarketSizing {
  // Assumption: a normalized addressable base of 1,000,000 units. Reach scales
  // with positive reception; capture scales with measured action-likelihood.
  const addressableBase = 1_000_000;
  const reachFactor = clamp(0.4 + 0.4 * positiveShare, 0.3, 0.85);
  const adoption = clamp(adoptionRate, 0.02, 0.95);
  const tam = addressableBase;
  const sam = Math.round(tam * reachFactor);
  const som = Math.round(sam * adoption);
  return { tam, sam, som, addressableBase, reachFactor, adoptionRate: adoption };
}

function deriveScenarios(forecast: ForecastReadyPayload | null): ScenarioStat[] {
  if (!forecast || !forecast.predictions.length) return [];
  const baseStart =
    forecast.history.at(-1)?.value ?? forecast.predictions[0].value;
  const base = forecast.predictions.at(-1)?.value ?? baseStart;
  const last = forecast.intervals.at(-1);
  const optimistic = last?.upper ?? base * 1.1;
  const pessimistic = last?.lower ?? base * 0.9;
  const change = (v: number) => (baseStart ? ((v - baseStart) / Math.abs(baseStart)) * 100 : 0);
  return [
    { name: "Pessimistic", endValue: pessimistic, pctChange: change(pessimistic) },
    { name: "Base", endValue: base, pctChange: change(base) },
    { name: "Optimistic", endValue: optimistic, pctChange: change(optimistic) },
  ];
}

function deriveSensitivity(causal: CausalGraphPayload | null): NamedWeight[] {
  if (!causal) return [];
  const labelOf = new Map(causal.nodes.map((n) => [n.id, n.label]));
  return causal.edges
    .map((e) => ({ label: labelOf.get(e.source) ?? e.source, weight: e.weight }))
    .sort((a, b) => b.weight - a.weight)
    .slice(0, 7);
}

function deriveMatrices(causal: CausalGraphPayload | null): {
  growth: MatrixPoint[];
  risk: MatrixPoint[];
} {
  if (!causal) return { growth: [], risk: [] };
  const drivers = causal.nodes.filter((n) => n.kind !== "goal").slice(0, 12);
  const growth = drivers.map((n) => ({
    label: n.label,
    x: n.criticality,
    y: n.prediction,
    size: n.criticality,
  }));
  const risk = drivers.map((n) => ({
    label: n.label,
    x: clamp((100 - n.prediction) / 100, 0, 1), // likelihood of shortfall
    y: clamp(n.criticality / 100, 0, 1), // impact
    size: n.criticality,
  }));
  return { growth, risk };
}

function deriveConcerns(opinions: PersonaOpinion[]): NamedWeight[] {
  const tally = new Map<string, number>();
  for (const o of opinions) {
    for (const c of o.key_concerns ?? []) {
      const key = c.trim();
      if (key) tally.set(key, (tally.get(key) ?? 0) + 1);
    }
  }
  return [...tally.entries()]
    .map(([label, weight]) => ({ label, weight }))
    .sort((a, b) => b.weight - a.weight)
    .slice(0, 8);
}

function deriveEvidenceBySource(evidence: EvidenceEntry[]): SourceBreakdownRow[] {
  const groups = new Map<string, { count: number; sentiments: number[]; titles: string[] }>();
  for (const e of evidence) {
    const g = groups.get(e.source) ?? { count: 0, sentiments: [], titles: [] };
    g.count += 1;
    if (typeof e.sentiment === "number") g.sentiments.push(e.sentiment);
    if (e.title) g.titles.push(e.title);
    groups.set(e.source, g);
  }
  return [...groups.entries()]
    .map(([source, g]) => ({
      source,
      count: g.count,
      meanSentiment: mean(g.sentiments),
      sampleTitles: g.titles.slice(0, 3),
    }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 10);
}

function deriveEvidenceFindings(evidence: EvidenceEntry[]): EvidenceFinding[] {
  const seen = new Set<string>();
  const rows: EvidenceFinding[] = [];
  for (const e of evidence) {
    const key = e.title.trim().toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    rows.push({
      source: e.source,
      title: e.title,
      detail: e.detail,
      url: e.url,
      sentiment: e.sentiment,
    });
  }
  return rows
    .sort((a, b) => Math.abs(b.sentiment ?? 0) - Math.abs(a.sentiment ?? 0))
    .slice(0, 12);
}

function deriveSwot(
  clusters: ClusterStat[],
  positiveShare: number,
  negativeShare: number,
  adoptionRate: number,
  forecastGrowthPct: number | null,
  concerns: NamedWeight[],
  ocean: OceanScores | null
): Swot {
  const strengths: string[] = [];
  const weaknesses: string[] = [];
  const opportunities: string[] = [];
  const threats: string[] = [];

  if (positiveShare > 0.4) strengths.push(`${pct(positiveShare)} of personas express positive sentiment.`);
  if (adoptionRate > 0.4) strengths.push(`High mean action-likelihood (${pct(adoptionRate)}).`);
  const enthusiasts = clusters.find((c) => /enthusiast/i.test(c.label));
  if (enthusiasts) strengths.push(`A committed "${enthusiasts.label}" segment (~${pct(enthusiasts.share)} of population).`);
  if (ocean && ocean.O > 60) strengths.push(`Audience skews high-Openness (${ocean.O.toFixed(0)}), receptive to novelty.`);
  if (!strengths.length) strengths.push("Balanced reception with no dominant negative signal.");

  if (negativeShare > 0.3) weaknesses.push(`${pct(negativeShare)} of personas are skeptical or negative.`);
  if (adoptionRate < 0.3) weaknesses.push(`Low conversion intent (action-likelihood ${pct(adoptionRate)}).`);
  if (ocean && ocean.N > 60) weaknesses.push(`Elevated Neuroticism (${ocean.N.toFixed(0)}) implies risk-averse hesitation.`);
  if (concerns[0]) weaknesses.push(`Leading concern: "${concerns[0].label}".`);
  if (!weaknesses.length) weaknesses.push("No acute structural weakness detected in the simulation.");

  if (forecastGrowthPct !== null && forecastGrowthPct > 0)
    opportunities.push(`Forecast projects ${forecastGrowthPct.toFixed(1)}% growth over the horizon.`);
  const pragmatists = clusters.find((c) => /pragmat/i.test(c.label));
  if (pragmatists) opportunities.push(`Convert the "${pragmatists.label}" middle (~${pct(pragmatists.share)}) with targeted proof points.`);
  opportunities.push("Tailor messaging per OCEAN cluster to lift conversion.");

  if (negativeShare > 0.25) threats.push(`Vocal skeptic segment could shape narrative (${pct(negativeShare)}).`);
  if (forecastGrowthPct !== null && forecastGrowthPct < 0)
    threats.push(`Forecast trend is negative (${forecastGrowthPct.toFixed(1)}%).`);
  if (concerns[1]) threats.push(`Secondary concern: "${concerns[1].label}".`);
  if (!threats.length) threats.push("Competitive response and execution risk remain the primary threats.");

  return { strengths, weaknesses, opportunities, threats };
}

function derivePorter(
  evidenceBySource: { source: string; count: number }[],
  concerns: NamedWeight[],
  adoptionRate: number,
  ocean: OceanScores | null
): PorterForce[] {
  const concernText = concerns.map((c) => c.label.toLowerCase()).join(" ");
  const priceSensitive = /price|cost|expensive|afford/.test(concernText);
  const rivalry = clamp(0.3 + evidenceBySource.length * 0.06, 0.2, 0.9);
  return [
    {
      force: "Competitive Rivalry",
      intensity: rivalry,
      note: `${evidenceBySource.length} distinct information sources in play.`,
    },
    {
      force: "Buyer Power",
      intensity: clamp(priceSensitive ? 0.75 : 0.45, 0.2, 0.9),
      note: priceSensitive ? "Price sensitivity surfaced in concerns." : "Moderate, value-driven buyers.",
    },
    {
      force: "Threat of Substitutes",
      intensity: clamp(ocean && ocean.O > 60 ? 0.65 : 0.45, 0.2, 0.9),
      note: ocean && ocean.O > 60 ? "High-Openness audience explores alternatives." : "Switching friction is moderate.",
    },
    {
      force: "Threat of New Entrants",
      intensity: 0.5,
      note: "Assumed moderate; no entry-barrier signal in simulation.",
    },
    {
      force: "Supplier Power",
      intensity: clamp(1 - adoptionRate * 0.4, 0.3, 0.7),
      note: "Assumed moderate; not directly observed.",
    },
  ];
}

function deriveRecommendations(
  clusters: ClusterStat[],
  positiveShare: number,
  adoptionRate: number,
  concerns: NamedWeight[],
  forecastGrowthPct: number | null
): Recommendation[] {
  const quick: string[] = [];
  const mid: string[] = [];
  const long: string[] = [];

  if (concerns[0]) quick.push(`Directly address the top concern ("${concerns[0].label}") in messaging.`);
  quick.push(`Target the highest-sentiment cluster${clusters[0] ? ` ("${clusters[0].label}")` : ""} for early wins.`);
  if (adoptionRate < 0.4) quick.push("Reduce friction in the activation path to lift action-likelihood.");

  mid.push("Stand up per-cluster messaging tracks aligned to OCEAN profiles.");
  if (positiveShare > 0.4) mid.push("Scale advocacy programs leveraging positive-sentiment personas.");
  mid.push("Instrument funnel metrics to validate the modeled adoption assumptions.");

  if (forecastGrowthPct !== null && forecastGrowthPct > 0)
    long.push("Invest ahead of the projected growth curve to capture share.");
  long.push("Build a defensible position against the strongest competitive force.");
  long.push("Re-run the simulation quarterly to track sentiment and adoption drift.");

  return [
    { horizon: "Quick Wins (0-3mo)", items: quick },
    { horizon: "Medium-Term (3-12mo)", items: mid },
    { horizon: "Long-Term (12mo+)", items: long },
  ];
}

export function buildReportModel(input: AnalyticsInput): ReportModel {
  const { personaOpinions: ops, forecast, causal } = input;
  const population = input.personasSimulated || ops.length;

  const sentiments = ops.map((o) => o.sentiment);
  const meanSentiment = mean(sentiments);
  const positiveShare = ops.length ? ops.filter((o) => o.sentiment > 0.15).length / ops.length : 0;
  const negativeShare = ops.length ? ops.filter((o) => o.sentiment < -0.15).length / ops.length : 0;
  const neutralShare = clamp(1 - positiveShare - negativeShare, 0, 1);
  const adoptionRate = mean(ops.map((o) => o.action_likelihood));

  const clusters = deriveClusters(ops, population);
  const marketSizing = deriveMarketSizing(positiveShare, adoptionRate);
  const scenarios = deriveScenarios(forecast);
  const sensitivity = deriveSensitivity(causal);
  const { growth, risk } = deriveMatrices(causal);
  const topConcerns = deriveConcerns(ops);
  const evidenceBySource = deriveEvidenceBySource(input.evidence);
  const evidenceFindings = deriveEvidenceFindings(input.evidence);

  let forecastGrowthPct: number | null = null;
  if (forecast && forecast.predictions.length) {
    const start = forecast.history.at(-1)?.value ?? forecast.predictions[0].value;
    const end = forecast.predictions.at(-1)?.value ?? start;
    forecastGrowthPct = start ? ((end - start) / Math.abs(start)) * 100 : null;
  }

  const swot = deriveSwot(
    clusters, positiveShare, negativeShare, adoptionRate, forecastGrowthPct, topConcerns, input.oceanMean
  );
  const porter = derivePorter(evidenceBySource, topConcerns, adoptionRate, input.oceanMean);
  const recommendations = deriveRecommendations(
    clusters, positiveShare, adoptionRate, topConcerns, forecastGrowthPct
  );

  const roiEstimate = adoptionRate * marketSizing.reachFactor * 100;
  const causalOutcome = causal?.overall_prediction ?? null;

  const kpis: Kpi[] = [
    {
      label: "Outcome Probability",
      value: causalOutcome === null ? "n/a" : `${causalOutcome.toFixed(1)}%`,
      hint: "blended simulation headline score",
      tone:
        causalOutcome === null
          ? "neutral"
          : causalOutcome >= 65
            ? "positive"
            : causalOutcome < 40
              ? "alert"
              : "neutral",
    },
    {
      label: "Population Simulated",
      value: population.toLocaleString(),
      hint: "IPIP-300 personas",
      tone: "neutral",
    },
    {
      label: "Mean Sentiment",
      value: meanSentiment.toFixed(2),
      hint: "-1 to +1 scale",
      tone: tone(meanSentiment),
    },
    {
      label: "Adoption Likelihood",
      value: pct(adoptionRate),
      hint: "mean action propensity",
      tone: adoptionRate > 0.4 ? "positive" : adoptionRate < 0.25 ? "alert" : "neutral",
    },
    {
      label: "Positive Share",
      value: pct(positiveShare),
      hint: "personas with +sentiment",
      tone: tone(positiveShare > 0 ? 0.2 : 0),
    },
    {
      label: "Forecast Growth",
      value: forecastGrowthPct === null ? "n/a" : `${forecastGrowthPct.toFixed(1)}%`,
      hint: forecast ? `${forecast.model} · ${forecast.horizon_days}d` : "no forecast",
      tone: forecastGrowthPct === null ? "neutral" : forecastGrowthPct >= 0 ? "positive" : "alert",
    },
    {
      label: "Est. ROI Index",
      value: roiEstimate.toFixed(0),
      hint: "model-derived (illustrative)",
      tone: roiEstimate > 30 ? "positive" : "neutral",
    },
    {
      label: "Serviceable (SOM)",
      value: marketSizing.som.toLocaleString(),
      hint: "est. capture, illustrative",
      tone: "neutral",
    },
    {
      label: "Forecast Accuracy",
      value: forecast ? forecast.mase_score.toFixed(2) : "n/a",
      hint: "MASE (lower is better)",
      tone: forecast && forecast.mase_score < 1 ? "positive" : "neutral",
    },
  ];

  const narrative = input.reportSections.map((s) => ({ title: s.title, content: s.content }));
  const appSection = input.reportSections.find((s) => s.title === "Simulation Applications");
  const extSection = input.reportSections.find((s) => s.title === "External Intelligence & Sources");
  const councilSection = input.reportSections.find((s) => s.title === "Council Consensus");
  const applicationPlaybooks = parseApplicationSection(appSection?.content ?? "");

  const deliberationMetrics: MetricBar[] | null = input.deliberation
    ? [
        { label: "Agreement", value: input.deliberation.agreement_rate * 100 },
        { label: "Polarization", value: input.deliberation.polarization_index * 100 },
        { label: "Contagion", value: input.deliberation.social_contagion_index * 100 },
        { label: "Entropy", value: input.deliberation.entropy_mean * 100 },
      ]
    : null;

  const consensusScores: ConsensusScores | null = input.consensus
    ? {
        agreement: input.consensus.agreement_score * 100,
        councilAlignment: input.consensus.council_alignment * 100,
        recommendedAction: input.consensus.recommended_action,
      }
    : null;

  const socialNarratives: SocialNarrativeRow[] = (input.socialSimulation?.final_narratives ?? []).map(
    (n) => ({
      label: n.label,
      adoptionPct: n.adoption_pct,
      sentiment: n.sentiment,
    })
  );

  const councilConfidence: CouncilConfidenceRow[] = (
    input.council?.opinions?.length
      ? input.council.opinions
      : input.councilOpinions
  ).map((op) => ({
    role: op.role || op.specialist_id,
    confidence: op.confidence * 100,
  }));

  return {
    query: input.query,
    generatedAt: new Date().toISOString(),
    population,
    sampleResponses: ops.length,
    meanSentiment,
    positiveShare,
    negativeShare,
    neutralShare,
    adoptionRate,
    oceanMean: input.oceanMean,
    forecastGrowthPct,
    forecastModel: forecast?.model ?? null,
    forecastMase: forecast?.mase_score ?? null,
    kpis,
    clusters,
    marketSizing,
    scenarios,
    sensitivity,
    growthMatrix: growth,
    riskMatrix: risk,
    swot,
    porter,
    topConcerns,
    evidenceBySource,
    evidenceFindings,
    externalIntelligenceContent: extSection?.content ?? null,
    councilConsensusContent: councilSection?.content ?? null,
    causalOutcome,
    deliberationMetrics,
    consensusScores,
    socialNarratives,
    councilConfidence,
    recommendations,
    narrative,
    applicationPlaybooks,
    heatmap: input.heatmap,
    personaPoints: input.personaPoints,
    causal,
    forecast,
  };
}
