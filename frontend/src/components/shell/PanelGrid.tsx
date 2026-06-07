import { memo } from "react";
import { Responsive, WidthProvider } from "react-grid-layout";
import { DashboardPanel } from "./DashboardPanel";
import {
  GRID_BREAKPOINTS,
  GRID_COLS_MAP,
  GRID_MARGIN,
  GRID_ROW_HEIGHT,
  usePanelLayouts,
} from "./panelLayout";
import { OCEANRadar } from "@/components/panels/OCEANRadar";
import { SentimentHeatmap } from "@/components/panels/SentimentHeatmap";
import { TimeSeriesPanel } from "@/components/panels/TimeSeriesPanel";
import { PersonaScatter3D } from "@/components/panels/PersonaScatter3D";
import { CausalMapD3 } from "@/components/panels/CausalMapD3";
import { EvidenceFeed } from "@/components/panels/EvidenceFeed";
import { PersonaOpinionsFeed } from "@/components/panels/PersonaOpinionsFeed";
import { SourceBreakdownChart } from "@/components/panels/SourceBreakdownChart";
import { PredictionOverview } from "@/components/panels/PredictionOverview";
import { SocialInteractionPanel } from "@/components/panels/SocialInteractionPanel";
import { CouncilPanel } from "@/components/panels/CouncilPanel";
import { ReportViewer } from "@/components/panels/ReportViewer";
import { useSessionStore } from "@/store/sessionStore";

const ResponsiveGridLayout = WidthProvider(Responsive);

function PersonasPanel() {
  const hasOpinions = useSessionStore((s) => s.personaOpinions.length > 0);
  return (
    <DashboardPanel label="World Environment // 1,500 Agents" code="PSY-OP" flush hasData={hasOpinions}>
      <PersonaOpinionsFeed />
    </DashboardPanel>
  );
}

function PredictionPanel() {
  const hasForecast = useSessionStore((s) => s.forecast !== null);
  const hasCausal = useSessionStore((s) => s.causal !== null);
  const hasDeliberation = useSessionStore((s) => s.deliberation !== null);
  const hasConsensus = useSessionStore((s) => s.consensus !== null);
  return (
    <DashboardPanel
      label="Prediction Overview"
      code="PRD-00"
      flush
      hasData={hasForecast || hasCausal || hasDeliberation || hasConsensus}
    >
      <PredictionOverview />
    </DashboardPanel>
  );
}

function ForecastPanel() {
  const hasData = useSessionStore((s) => s.forecast !== null);
  return (
    <DashboardPanel label="Forecast // TimesFM-ICF" code="FCT-05" flush hasData={hasData}>
      <TimeSeriesPanel />
    </DashboardPanel>
  );
}

function OceanPanel() {
  const hasData = useSessionStore((s) => s.oceanMean !== null);
  return (
    <DashboardPanel label="OCEAN Distribution" code="PSY-04" flush hasData={hasData}>
      <OCEANRadar />
    </DashboardPanel>
  );
}

function ScatterPanel() {
  const hasData = useSessionStore((s) => s.personaPoints.length > 0);
  return (
    <DashboardPanel label="Persona PCA Space" code="PSY-3D" flush hasData={hasData}>
      <PersonaScatter3D />
    </DashboardPanel>
  );
}

function HeatmapPanel() {
  const hasData = useSessionStore((s) => s.heatmap.length > 0);
  return (
    <DashboardPanel label="Sentiment Heatmap" code="PSY-HM" flush hasData={hasData}>
      <SentimentHeatmap />
    </DashboardPanel>
  );
}

function EvidencePanel() {
  const hasData = useSessionStore((s) => s.evidence.length > 0);
  return (
    <DashboardPanel label="Evidence Feed" code="EVD-02" flush hasData={hasData}>
      <EvidenceFeed />
    </DashboardPanel>
  );
}

function SourcesPanel() {
  const hasData = useSessionStore((s) => s.evidence.length > 0);
  return (
    <DashboardPanel label="Web Source Breakdown" code="EVD-SB" flush hasData={hasData}>
      <SourceBreakdownChart />
    </DashboardPanel>
  );
}

function CausalPanel() {
  const hasData = useSessionStore((s) => s.causal !== null);
  return (
    <DashboardPanel label="Causal Mapping" code="CSL-09" flush hasData={hasData}>
      <CausalMapD3 />
    </DashboardPanel>
  );
}

function SocialPanel() {
  const hasData = useSessionStore(
    (s) => s.socialTicks.length > 0 || s.socialSimulation !== null
  );
  return (
    <DashboardPanel label="Social Interaction" code="SOC-01" flush hasData={hasData}>
      <SocialInteractionPanel />
    </DashboardPanel>
  );
}

function CouncilPanelTile() {
  const hasData = useSessionStore(
    (s) => s.councilOpinions.length > 0 || s.council !== null
  );
  return (
    <DashboardPanel label="Specialist Council" code="CNC-04" flush hasData={hasData}>
      <CouncilPanel />
    </DashboardPanel>
  );
}

function ReportPanel() {
  const hasData = useSessionStore((s) => s.reportSections.length > 0);
  return (
    <DashboardPanel label="Strategic Report" code="RPT-07" flush hasData={hasData}>
      <ReportViewer />
    </DashboardPanel>
  );
}

const MemoPersonas = memo(PersonasPanel);
const MemoPrediction = memo(PredictionPanel);
const MemoForecast = memo(ForecastPanel);
const MemoOcean = memo(OceanPanel);
const MemoScatter = memo(ScatterPanel);
const MemoHeatmap = memo(HeatmapPanel);
const MemoEvidence = memo(EvidencePanel);
const MemoSources = memo(SourcesPanel);
const MemoCausal = memo(CausalPanel);
const MemoSocial = memo(SocialPanel);
const MemoCouncil = memo(CouncilPanelTile);

export function PanelGrid() {
  const { layouts, handleLayoutChange, resetLayouts } = usePanelLayouts();

  return (
    <div className="panel-grid-scroll relative min-h-0 flex-1 overflow-y-auto overflow-x-hidden px-2 pb-2">
      <div className="sticky top-0 z-10 -mx-2 mb-1 flex items-center justify-between bg-bg/80 px-2 py-1 backdrop-blur-sm">
        <span className="font-mono text-2xs uppercase tracking-widest text-muted">
          Drag headers to move · drag edges to resize · expand for full view
        </span>
        <button
          type="button"
          onClick={resetLayouts}
          className="rounded border border-[color:var(--hairline)] px-2 py-0.5 font-mono text-2xs uppercase tracking-widest text-muted transition-colors hover:border-teal hover:text-teal"
          title="Restore the default tile arrangement"
        >
          Reset layout
        </button>
      </div>

      <ResponsiveGridLayout
        className="layout"
        layouts={layouts}
        breakpoints={GRID_BREAKPOINTS}
        cols={GRID_COLS_MAP}
        rowHeight={GRID_ROW_HEIGHT}
        margin={GRID_MARGIN}
        containerPadding={[0, 0]}
        draggableHandle=".panel-drag-handle"
        draggableCancel=".panel-no-drag"
        resizeHandles={["s", "e", "se", "sw"]}
        compactType="vertical"
        useCSSTransforms
        onLayoutChange={handleLayoutChange}
      >
        <div key="agents" className="panel-grid-tile">
          <MemoPersonas />
        </div>
        <div key="prediction" className="panel-grid-tile">
          <MemoPrediction />
        </div>
        <div key="forecast" className="panel-grid-tile">
          <MemoForecast />
        </div>
        <div key="ocean" className="panel-grid-tile">
          <MemoOcean />
        </div>
        <div key="scatter" className="panel-grid-tile">
          <MemoScatter />
        </div>
        <div key="heatmap" className="panel-grid-tile">
          <MemoHeatmap />
        </div>
        <div key="evidence" className="panel-grid-tile">
          <MemoEvidence />
        </div>
        <div key="sources" className="panel-grid-tile">
          <MemoSources />
        </div>
        <div key="causal" className="panel-grid-tile">
          <MemoCausal />
        </div>
        <div key="social" className="panel-grid-tile">
          <MemoSocial />
        </div>
        <div key="council" className="panel-grid-tile">
          <MemoCouncil />
        </div>
        <div key="report" className="panel-grid-tile">
          <ReportPanel />
        </div>
      </ResponsiveGridLayout>
    </div>
  );
}
