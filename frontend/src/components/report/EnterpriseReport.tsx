import { useMemo, useRef, useState } from "react";
import { useSessionStore } from "@/store/sessionStore";
import { buildReportModel, DISCLAIMER } from "@/lib/reportAnalytics";
import { USE_CASES } from "@/lib/useCases";
import { renderMarkdown } from "./markdown";
import { captureReport } from "./exporters/capture";
import { exportReportToPdf } from "./exporters/pdfExport";
import { exportReportToWord } from "./exporters/wordExport";
import { exportReportToPptx } from "./exporters/pptxExport";
import type { ReportExportInput } from "./exporters/types";
import {
  AdoptionFunnelChart,
  CausalOutcomeChart,
  ClusterDonutChart,
  ClusterSentimentChart,
  ConcernsChart,
  ConsensusGaugeChart,
  CouncilConfidenceChart,
  DeliberationMetricsChart,
  EvidenceSentimentChart,
  EvidenceSourceChart,
  ForecastChart,
  GrowthMatrixChart,
  MarketSizingChart,
  OceanRadarChart,
  PersonaScatter3DChart,
  RiskMatrixChart,
  ScenarioChart,
  SensitivityTornadoChart,
  SentimentDistributionChart,
  SentimentHeatmapChart,
  SocialNarrativeChart,
} from "./charts/ReportCharts";
import {
  KpiGrid,
  MarketAssumptions,
  PorterForces,
  RecommendationsRoadmap,
  SwotMatrix,
} from "./frameworks/Frameworks";

import type { EvidenceFinding, ReportModel } from "@/lib/reportAnalytics";

type ExportKind = "pdf" | "docx" | "pptx";

const NARRATIVE_EXCLUDE = new Set([
  "Simulation Applications",
  "External Intelligence & Sources",
  "Council Consensus",
]);

function SectionTitle({ children, accent = "teal" }: { children: string; accent?: "teal" | "orange" }) {
  return (
    <h3
      className={`mt-2 border-l-2 pl-2 font-display text-sm font-bold uppercase tracking-wider text-data ${
        accent === "orange" ? "border-orange" : "border-teal"
      }`}
    >
      {children}
    </h3>
  );
}

function sentimentLabel(sent: number | undefined): string {
  if (sent == null) return "—";
  if (sent >= 0.15) return "Positive";
  if (sent <= -0.15) return "Negative";
  return "Neutral";
}

function ExternalIntelligenceSection({ model }: { model: ReportModel }) {
  const hasBackend = Boolean(model.externalIntelligenceContent);
  const hasFindings = model.evidenceFindings.length > 0;

  if (!hasBackend && !hasFindings) return null;

  return (
    <>
      <SectionTitle accent="orange">External Intelligence & Sources</SectionTitle>
      <div className="space-y-3">
        {hasBackend && (
          <article className="rounded-md border border-[color:var(--hairline)] bg-panel/40 p-3">
            <div className="space-y-1.5">{renderMarkdown(model.externalIntelligenceContent!)}</div>
          </article>
        )}
        {hasFindings && (
          <div className="overflow-x-auto rounded-md border border-[color:var(--hairline)] bg-panel/30">
            <table className="w-full min-w-[28rem] border-collapse font-mono text-2xs">
              <thead>
                <tr className="border-b border-[color:var(--hairline)] text-left uppercase tracking-wider text-muted">
                  <th className="px-2 py-1.5 font-semibold">Source</th>
                  <th className="px-2 py-1.5 font-semibold">Finding</th>
                  <th className="px-2 py-1.5 font-semibold">Sent.</th>
                  <th className="px-2 py-1.5 font-semibold">Link</th>
                </tr>
              </thead>
              <tbody>
                {model.evidenceFindings.map((f: EvidenceFinding, i) => (
                  <tr key={i} className="border-b border-[color:var(--hairline)]/60 text-data/90">
                    <td className="px-2 py-1.5 align-top text-teal">{f.source}</td>
                    <td className="px-2 py-1.5 align-top">
                      <span className="font-semibold text-data">{f.title}</span>
                      {f.detail && (
                        <p className="mt-0.5 leading-relaxed text-muted">{f.detail.slice(0, 140)}</p>
                      )}
                    </td>
                    <td className="px-2 py-1.5 align-top text-muted">{sentimentLabel(f.sentiment)}</td>
                    <td className="px-2 py-1.5 align-top">
                      {f.url ? (
                        <a
                          href={f.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-teal underline-offset-2 hover:underline"
                        >
                          source
                        </a>
                      ) : (
                        <span className="text-muted">—</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {(model.evidenceBySource.length > 0 ||
          model.evidenceFindings.some((f) => f.sentiment != null)) && (
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
            {model.evidenceBySource.length > 0 && <EvidenceSourceChart model={model} />}
            {model.evidenceFindings.some((f) => f.sentiment != null) && (
              <EvidenceSentimentChart model={model} />
            )}
          </div>
        )}
      </div>
    </>
  );
}

function CouncilConsensusSection({ model }: { model: ReportModel }) {
  const hasCharts = model.councilConfidence.length > 0;
  const hasContent = Boolean(model.councilConsensusContent);
  if (!hasCharts && !hasContent) return null;

  return (
    <>
      <SectionTitle accent="orange">Council & Consensus</SectionTitle>
      {hasCharts && (
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
          <CouncilConfidenceChart model={model} />
        </div>
      )}
      {hasContent && (
        <article className="rounded-md border border-[color:var(--hairline)] bg-panel/40 p-3">
          <div className="space-y-1.5">{renderMarkdown(model.councilConsensusContent!)}</div>
        </article>
      )}
    </>
  );
}

function SimulationApplicationsGrid({
  playbooks,
}: {
  playbooks: ReturnType<typeof buildReportModel>["applicationPlaybooks"];
}) {
  const cards = playbooks.length >= USE_CASES.length ? playbooks : USE_CASES.map((uc) => {
    const found = playbooks.find((p) => p.domain === uc.domain);
    return (
      found ?? {
        domain: uc.domain,
        tagline: uc.tagline,
        description: uc.description,
        simulationInsight: "",
      }
    );
  });

  return (
    <div className="grid grid-cols-1 gap-3 md:grid-cols-2 lg:grid-cols-3">
      {cards.map((pb) => (
        <article
          key={pb.domain}
          className="flex flex-col gap-2 rounded-md border border-[color:var(--hairline)] bg-panel/40 p-3"
        >
          <div>
            <h4 className="font-display text-xs font-bold uppercase tracking-wider text-teal">
              {pb.domain}
            </h4>
            <p className="font-mono text-2xs uppercase tracking-widest text-orange">{pb.tagline}</p>
          </div>
          <p className="font-mono text-2xs leading-relaxed text-muted">{pb.description}</p>
          {pb.simulationInsight && (
            <p className="mt-auto border-t border-[color:var(--hairline)] pt-2 font-mono text-2xs leading-relaxed text-data">
              <span className="text-teal">Simulation insight:</span> {pb.simulationInsight}
            </p>
          )}
          {pb.recommendedAction && (
            <p className="font-mono text-2xs leading-relaxed text-data/90">
              <span className="text-orange">Recommended:</span> {pb.recommendedAction}
            </p>
          )}
        </article>
      ))}
    </div>
  );
}

export function EnterpriseReport({ onClose }: { onClose: () => void }) {
  const rootQuery = useSessionStore((s) => s.rootQuery);
  const evidence = useSessionStore((s) => s.evidence);
  const oceanMean = useSessionStore((s) => s.oceanMean);
  const personaPoints = useSessionStore((s) => s.personaPoints);
  const personaOpinions = useSessionStore((s) => s.personaOpinions);
  const heatmap = useSessionStore((s) => s.heatmap);
  const personasSimulated = useSessionStore((s) => s.personasSimulated);
  const forecast = useSessionStore((s) => s.forecast);
  const causal = useSessionStore((s) => s.causal);
  const deliberation = useSessionStore((s) => s.deliberation);
  const consensus = useSessionStore((s) => s.consensus);
  const socialSimulation = useSessionStore((s) => s.socialSimulation);
  const council = useSessionStore((s) => s.council);
  const councilOpinions = useSessionStore((s) => s.councilOpinions);
  const reportSections = useSessionStore((s) => s.reportSections);
  const pushToast = useSessionStore((s) => s.pushToast);

  const reportRef = useRef<HTMLDivElement>(null);
  const [exporting, setExporting] = useState<ExportKind | null>(null);

  const model = useMemo(
    () =>
      buildReportModel({
        query: rootQuery,
        evidence,
        oceanMean,
        personaPoints,
        personaOpinions,
        heatmap,
        personasSimulated,
        forecast,
        causal,
        reportSections,
        deliberation,
        consensus,
        socialSimulation,
        council,
        councilOpinions,
      }),
    [
      rootQuery,
      evidence,
      oceanMean,
      personaPoints,
      personaOpinions,
      heatmap,
      personasSimulated,
      forecast,
      causal,
      reportSections,
      deliberation,
      consensus,
      socialSimulation,
      council,
      councilOpinions,
    ]
  );

  const runExport = async (kind: ExportKind) => {
    if (!reportRef.current || exporting) return;
    setExporting(kind);
    try {
      const blocks = await captureReport(reportRef.current);
      const input: ReportExportInput = {
        title: "Project Singularity - Strategic Report",
        query: model.query,
        generatedAt: model.generatedAt,
        narrative: model.narrative,
        blocks,
        disclaimer: DISCLAIMER,
      };
      if (kind === "pdf") await exportReportToPdf(input);
      else if (kind === "docx") await exportReportToWord(input);
      else await exportReportToPptx(input);
      pushToast({ kind: "info", message: `${kind.toUpperCase()} export ready.` });
    } catch (err) {
      pushToast({
        kind: "error",
        message: err instanceof Error ? err.message : `${kind.toUpperCase()} export failed.`,
      });
    } finally {
      setExporting(null);
    }
  };

  const ExportBtn = ({ kind, label, color }: { kind: ExportKind; label: string; color: string }) => (
    <button
      type="button"
      onClick={() => void runExport(kind)}
      disabled={exporting !== null}
      className={`rounded-sm border px-2.5 py-1 font-mono text-2xs font-semibold uppercase tracking-wider transition-colors disabled:opacity-40 ${color}`}
    >
      {exporting === kind ? "Exporting..." : label}
    </button>
  );

  return (
    <div className="fixed inset-0 z-50 flex flex-col bg-bg/97 backdrop-blur-sm">
      <header className="flex shrink-0 items-center justify-between gap-3 border-b border-[color:var(--hairline)] px-4 py-3">
        <div className="min-w-0">
          <h2 className="truncate font-display text-sm font-bold uppercase tracking-wider text-data">
            Strategic Intelligence Report
          </h2>
          {model.query && <p className="truncate font-mono text-2xs text-muted">{model.query}</p>}
        </div>
        <div className="flex shrink-0 items-center gap-1.5">
          <ExportBtn kind="pdf" label="Export PDF" color="border-orange/50 bg-orange/10 text-orange hover:bg-orange/20" />
          <ExportBtn kind="docx" label="Export Word" color="border-teal/50 bg-teal/10 text-teal hover:bg-teal/20" />
          <ExportBtn kind="pptx" label="Export PPTX" color="border-positive/50 bg-positive/10 text-positive hover:bg-positive/20" />
          <button
            type="button"
            onClick={onClose}
            disabled={exporting !== null}
            className="rounded-sm border border-[color:var(--hairline)] px-2.5 py-1 font-mono text-2xs uppercase tracking-wider text-muted transition-colors hover:text-data disabled:opacity-40"
          >
            Close
          </button>
        </div>
      </header>

      {exporting && (
        <div className="shrink-0 bg-teal/10 px-4 py-1 font-mono text-2xs text-teal">
          Rendering charts to images and composing {exporting.toUpperCase()} - this can take a few seconds...
        </div>
      )}

      <div ref={reportRef} className="min-h-0 flex-1 overflow-auto bg-bg px-4 py-5">
        <div className="mx-auto max-w-6xl space-y-5">
          <SectionTitle>Executive Summary</SectionTitle>
          <KpiGrid model={model} />

          <ExternalIntelligenceSection model={model} />

          <SectionTitle>Simulation Intelligence</SectionTitle>
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
            <CausalOutcomeChart model={model} />
            <DeliberationMetricsChart model={model} />
            <ConsensusGaugeChart model={model} />
            <SocialNarrativeChart model={model} />
          </div>

          <SectionTitle>Behavioral Analysis</SectionTitle>
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
            <SentimentDistributionChart model={model} />
            <OceanRadarChart model={model} />
            <ClusterDonutChart model={model} />
            <ClusterSentimentChart model={model} />
            <AdoptionFunnelChart model={model} />
            <ConcernsChart model={model} />
            <SentimentHeatmapChart model={model} />
            <PersonaScatter3DChart model={model} />
          </div>

          <SectionTitle accent="orange">Market & Forecast</SectionTitle>
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
            <ForecastChart model={model} />
            <MarketSizingChart model={model} />
            <ScenarioChart model={model} />
          </div>
          <MarketAssumptions model={model} />

          <SectionTitle>Strategic Frameworks</SectionTitle>
          <SwotMatrix model={model} />
          <PorterForces model={model} />
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
            <GrowthMatrixChart model={model} />
            <RiskMatrixChart model={model} />
            <SensitivityTornadoChart model={model} />
          </div>
          <RecommendationsRoadmap model={model} />

          <CouncilConsensusSection model={model} />

          <SectionTitle accent="orange">Simulation Applications</SectionTitle>
          <SimulationApplicationsGrid playbooks={model.applicationPlaybooks} />

          {model.narrative.filter((s) => !NARRATIVE_EXCLUDE.has(s.title)).length > 0 && (
            <>
              <SectionTitle accent="orange">Narrative Analysis</SectionTitle>
              <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                {model.narrative
                  .filter((s) => !NARRATIVE_EXCLUDE.has(s.title))
                  .map((s, i) => (
                  <article key={i} className="space-y-1.5 rounded-md border border-[color:var(--hairline)] bg-panel/40 p-3">
                    <h4 className="mb-1 font-display text-xs font-bold uppercase tracking-wider text-teal">
                      {s.title}
                    </h4>
                    <div className="space-y-1.5">{renderMarkdown(s.content)}</div>
                  </article>
                ))}
              </div>
            </>
          )}

          <p className="border-t border-[color:var(--hairline)] pt-3 font-mono text-2xs leading-relaxed text-muted/70">
            {DISCLAIMER}
          </p>
        </div>
      </div>
    </div>
  );
}
