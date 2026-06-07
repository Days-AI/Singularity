import { useCallback, useEffect, useState } from "react";
import { HeaderBar } from "@/components/shell/HeaderBar";
import { QueryBar } from "@/components/shell/QueryBar";
import { PanelGrid } from "@/components/shell/PanelGrid";
import { MetricsStrip } from "@/components/shell/MetricsStrip";
import { Toasts } from "@/components/shell/Toasts";
import { DEFAULT_QUERY, useSSEStream } from "@/hooks/useSSEStream";
import { getStreamMode, generateReport } from "@/api/singularity";
import { MOCK_REPORT_SECTIONS } from "@/mock/scenario";
import { useSessionStore } from "@/store/sessionStore";

export default function App() {
  const { start, stop } = useSSEStream();
  const [query, setQuery] = useState(DEFAULT_QUERY);
  const [questions, setQuestions] = useState<string[]>([]);
  const [reportLoading, setReportLoading] = useState(false);

  const connection = useSessionStore((s) => s.connection);
  const setReportSections = useSessionStore((s) => s.setReportSections);
  const pushToast = useSessionStore((s) => s.pushToast);

  const isRunning = connection === "connecting" || connection === "streaming";

  const run = useCallback(() => {
    const trimmed = questions.map((q) => q.trim()).filter(Boolean);
    void start(query.trim() || DEFAULT_QUERY, trimmed);
  }, [query, questions, start]);

  // Auto-run a simulation on first mount so the terminal is never empty.
  useEffect(() => {
    void start(DEFAULT_QUERY);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleGenerateReport = useCallback(async () => {
    const trimmed = questions.map((q) => q.trim()).filter(Boolean);
    setReportLoading(true);
    try {
      if (getStreamMode() === "mock") {
        setReportSections(MOCK_REPORT_SECTIONS);
        return;
      }
      const sessionId = useSessionStore.getState().sessionId;
      if (!sessionId) {
        pushToast({ kind: "error", message: "No completed session to report on yet." });
        return;
      }
      const sections = await generateReport(sessionId, trimmed);
      setReportSections(sections);
    } catch (err) {
      pushToast({
        kind: "error",
        message: err instanceof Error ? err.message : "Report generation failed.",
      });
    } finally {
      setReportLoading(false);
    }
  }, [questions, setReportSections, pushToast]);

  return (
    <div className="fx-grain fx-scan flex h-screen w-screen flex-col overflow-hidden bg-bg">
      <HeaderBar onRun={run} onStop={stop} />
      <QueryBar
        query={query}
        onQueryChange={setQuery}
        questions={questions}
        onQuestionsChange={setQuestions}
        disabled={isRunning}
        onRun={run}
      />
      <main className="flex min-h-0 flex-1 flex-col">
        <PanelGrid
          onGenerateReport={handleGenerateReport}
          reportLoading={reportLoading}
        />
      </main>
      <MetricsStrip />
      <Toasts />
    </div>
  );
}
