import { useCallback, useEffect, useRef, useState } from "react";
import { HeaderBar } from "@/components/shell/HeaderBar";
import { QueryBar } from "@/components/shell/QueryBar";
import { PanelGrid } from "@/components/shell/PanelGrid";
import { MetricsStrip } from "@/components/shell/MetricsStrip";
import { Toasts } from "@/components/shell/Toasts";
import { DEFAULT_QUERY, useSSEStream } from "@/hooks/useSSEStream";
import { fetchHealth } from "@/api/singularity";
import { useSessionStore } from "@/store/sessionStore";

export default function App() {
  const { start, stop } = useSSEStream();
  const [query, setQuery] = useState(DEFAULT_QUERY);
  const [questions, setQuestions] = useState<string[]>([]);
  const [webSourcesEnabled, setWebSourcesEnabled] = useState(true);
  const connection = useSessionStore((s) => s.connection);
  const pushToast = useSessionStore((s) => s.pushToast);

  // Only lock inputs during active SSE delivery; connecting can take minutes on CPU.
  const inputsLocked = connection === "streaming";

  const run = useCallback(() => {
    const trimmed = questions.map((q) => q.trim()).filter(Boolean);
    void start(query.trim() || DEFAULT_QUERY, trimmed, webSourcesEnabled);
  }, [query, questions, webSourcesEnabled, start]);

  // Auto-run a simulation on first mount so the terminal is never empty.
  // Guarded so React StrictMode's double-invoked mount effect (dev) cannot
  // start two concurrent flows / EventSources, which raced the stream and
  // duplicated backend sessions.
  const didAutoRun = useRef(false);
  useEffect(() => {
    if (didAutoRun.current) return;
    didAutoRun.current = true;
    void (async () => {
      const health = await fetchHealth();
      if (!health) {
        pushToast({
          kind: "error",
          message:
            "Backend not running on :8000 — Ollama/Gemma will not be used. Run run.bat or start uvicorn in backend/.",
        });
      } else if (!health.ollama.reachable) {
        pushToast({
          kind: "error",
          message: `Ollama unreachable at ${health.ollama.base_url}. Start Ollama, then: ollama pull ${health.ollama_model}`,
        });
      } else if (!health.ollama.model_available) {
        pushToast({
          kind: "error",
          message: `Ollama model '${health.ollama_model}' not found. Run: ollama pull ${health.ollama_model}`,
        });
      }
      void start(DEFAULT_QUERY);
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="fx-grain fx-scan flex h-screen w-screen flex-col overflow-hidden bg-bg">
      <HeaderBar onRun={run} onStop={stop} />
      <QueryBar
        query={query}
        onQueryChange={setQuery}
        questions={questions}
        onQuestionsChange={setQuestions}
        webSourcesEnabled={webSourcesEnabled}
        onToggleWebSources={setWebSourcesEnabled}
        disabled={inputsLocked}
        onRun={run}
      />
      <main className="flex min-h-0 flex-1 flex-col">
        <PanelGrid />
      </main>
      <MetricsStrip />
      <Toasts />
    </div>
  );
}
