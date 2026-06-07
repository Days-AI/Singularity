import { useEffect, useState } from "react";
import { useSessionStore, type ConnectionStatus } from "@/store/sessionStore";
import { fmtClock, fmtDuration } from "@/lib/format";

const STATUS_META: Record<
  ConnectionStatus,
  { label: string; cls: string }
> = {
  idle: { label: "STANDBY", cls: "text-muted" },
  connecting: { label: "LINKING", cls: "text-orange" },
  streaming: { label: "LIVE", cls: "text-teal" },
  complete: { label: "RESOLVED", cls: "text-positive" },
  error: { label: "FAULT", cls: "text-alert" },
};

interface HeaderBarProps {
  onRun: () => void;
  onStop: () => void;
}

export function HeaderBar({ onRun, onStop }: HeaderBarProps) {
  const [now, setNow] = useState(() => new Date());
  const connection = useSessionStore((s) => s.connection);
  const activeAgents = useSessionStore((s) => s.activeAgents);
  const startedAt = useSessionStore((s) => s.startedAt);
  const durationMs = useSessionStore((s) => s.durationMs);
  const rootQuery = useSessionStore((s) => s.rootQuery);

  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(id);
  }, []);

  const meta = STATUS_META[connection];
  const isRunning = connection === "streaming" || connection === "connecting";
  const elapsed =
    durationMs ?? (startedAt ? Date.now() - startedAt : null);

  return (
    <header className="flex h-12 shrink-0 items-center gap-4 border-b border-[color:var(--hairline)] bg-panel px-4">
      <div className="flex items-baseline gap-2">
        <span className="font-display text-base font-extrabold uppercase tracking-[0.22em] text-data">
          Singularity
        </span>
        <span className="font-mono text-2xs text-muted">v0.1.0</span>
      </div>

      <div className="hidden items-center gap-1.5 md:flex">
        <span
          className={`h-2 w-2 rounded-full ${
            isRunning ? "animate-pulse-stream" : ""
          } ${meta.cls.replace("text-", "bg-")}`}
        />
        <span className={`font-mono text-xs font-semibold ${meta.cls}`}>
          {meta.label}
        </span>
      </div>

      <div className="hidden min-w-0 flex-1 truncate font-mono text-xs text-muted lg:block">
        {rootQuery ? (
          <span>
            <span className="text-muted/60">QUERY //</span>{" "}
            <span className="text-data">{rootQuery}</span>
          </span>
        ) : (
          <span className="text-muted/60">awaiting simulation query</span>
        )}
      </div>

      <div className="ml-auto flex items-center gap-5">
        <Metric label="AGENTS" value={String(activeAgents).padStart(2, "0")} live={activeAgents > 0} />
        <Metric label="ELAPSED" value={fmtDuration(elapsed)} />
        <Metric label="UTC" value={fmtClock(now)} />

        {isRunning ? (
          <button
            onClick={onStop}
            className="rounded-sm border border-alert/50 px-3 py-1 font-mono text-xs font-semibold uppercase tracking-wider text-alert transition-colors hover:bg-alert/10"
          >
            Abort
          </button>
        ) : (
          <button
            onClick={onRun}
            className="rounded-sm border border-teal/50 bg-teal/10 px-3 py-1 font-mono text-xs font-semibold uppercase tracking-wider text-teal shadow-glow transition-colors hover:bg-teal/20"
          >
            Run Sim
          </button>
        )}
      </div>
    </header>
  );
}

function Metric({
  label,
  value,
  live = false,
}: {
  label: string;
  value: string;
  live?: boolean;
}) {
  return (
    <div className="flex flex-col items-end leading-none">
      <span className="text-2xs uppercase tracking-widest text-muted">
        {label}
      </span>
      <span
        className={`stat-value text-sm ${live ? "text-teal" : "text-data"}`}
      >
        {value}
      </span>
    </div>
  );
}
