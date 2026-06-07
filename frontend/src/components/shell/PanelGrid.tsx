import { Panel, type PanelStatus } from "./Panel";
import { DAGVisualizerD3 } from "@/components/panels/DAGVisualizerD3";
import { OCEANRadar } from "@/components/panels/OCEANRadar";
import { SentimentHeatmap } from "@/components/panels/SentimentHeatmap";
import { TimeSeriesPanel } from "@/components/panels/TimeSeriesPanel";
import { PersonaScatter3D } from "@/components/panels/PersonaScatter3D";
import { CausalSankeyD3 } from "@/components/panels/CausalSankeyD3";
import { EvidenceFeed } from "@/components/panels/EvidenceFeed";
import { ReportViewer } from "@/components/panels/ReportViewer";
import { useSessionStore } from "@/store/sessionStore";

/** Maps a boolean "has data" + connection into a panel status indicator. */
function useStatus(hasData: boolean): PanelStatus {
  const connection = useSessionStore((s) => s.connection);
  if (connection === "error") return hasData ? "done" : "error";
  if (hasData) return connection === "complete" ? "done" : "live";
  return connection === "streaming" || connection === "connecting"
    ? "live"
    : "idle";
}

interface PanelGridProps {
  onGenerateReport: () => void;
  reportLoading: boolean;
}

export function PanelGrid({ onGenerateReport, reportLoading }: PanelGridProps) {
  const hasDag = useSessionStore((s) => s.dagNodes.length > 0);
  const hasOcean = useSessionStore((s) => s.oceanMean !== null);
  const hasHeatmap = useSessionStore((s) => s.heatmap.length > 0);
  const hasForecast = useSessionStore((s) => s.forecast !== null);
  const hasScatter = useSessionStore((s) => s.personaPoints.length > 0);
  const hasCausal = useSessionStore((s) => s.causal !== null);
  const hasEvidence = useSessionStore((s) => s.evidence.length > 0);
  const hasReport = useSessionStore((s) => s.reportSections.length > 0);

  return (
    <div className="panel-grid min-h-0 flex-1 p-2">
      <Panel
        label="DAG Execution Graph"
        code="DAG-01"
        status={useStatus(hasDag)}
        order={0}
        flush
        className="area-dag"
      >
        <DAGVisualizerD3 />
      </Panel>

      <Panel
        label="Forecast // TimesFM-ICF"
        code="FCT-05"
        status={useStatus(hasForecast)}
        order={1}
        flush
        className="area-forecast"
      >
        <TimeSeriesPanel />
      </Panel>

      <Panel
        label="OCEAN Distribution"
        code="PSY-04"
        status={useStatus(hasOcean)}
        order={2}
        flush
        className="area-ocean"
      >
        <OCEANRadar />
      </Panel>

      <Panel
        label="Persona PCA Space"
        code="PSY-3D"
        status={useStatus(hasScatter)}
        order={3}
        flush
        className="area-scatter"
      >
        <PersonaScatter3D />
      </Panel>

      <Panel
        label="Sentiment Heatmap"
        code="PSY-HM"
        status={useStatus(hasHeatmap)}
        order={4}
        flush
        className="area-heatmap"
      >
        <SentimentHeatmap />
      </Panel>

      <Panel
        label="Evidence Feed"
        code="EVD-02"
        status={useStatus(hasEvidence)}
        order={5}
        flush
        className="area-evidence"
      >
        <EvidenceFeed />
      </Panel>

      <Panel
        label="Causal Inference Map"
        code="CSL-09"
        status={useStatus(hasCausal)}
        order={6}
        flush
        className="area-causal"
      >
        <CausalSankeyD3 />
      </Panel>

      <Panel
        label="Strategic Report"
        code="RPT-07"
        status={useStatus(hasReport)}
        order={7}
        className="area-report"
      >
        <ReportViewer
          onGenerateReport={onGenerateReport}
          reportLoading={reportLoading}
        />
      </Panel>
    </div>
  );
}
