import type { ReactNode } from "react";

export type PanelStatus = "idle" | "live" | "done" | "error";

interface PanelProps {
  label: string;
  /** Short code shown on the right of the header, e.g. "DAG-01". */
  code?: string;
  status?: PanelStatus;
  /** Stagger index for the orchestrated load-in animation. */
  order?: number;
  className?: string;
  /** When true the body becomes a non-scrolling flex container. */
  flush?: boolean;
  children: ReactNode;
}

const STATUS_DOT: Record<PanelStatus, string> = {
  idle: "bg-muted",
  live: "bg-teal animate-pulse-stream",
  done: "bg-positive",
  error: "bg-alert",
};

export function Panel({
  label,
  code,
  status = "idle",
  order = 0,
  className = "",
  flush = false,
  children,
}: PanelProps) {
  return (
    <section
      className={`group flex min-h-0 flex-col rounded-[var(--panel-radius)] border border-[color:var(--hairline)] bg-panel shadow-panel animate-panel-in ${className}`}
      style={{ animationDelay: `${order * 70}ms` }}
    >
      <header className="flex items-center justify-between border-b border-[color:var(--hairline)] px-3 py-1.5">
        <div className="flex items-center gap-2">
          <span className={`h-1.5 w-1.5 rounded-full ${STATUS_DOT[status]}`} />
          <h2 className="panel-label">{label}</h2>
        </div>
        {code && (
          <span className="font-mono text-2xs tracking-widest text-muted">
            {code}
          </span>
        )}
      </header>
      <div
        className={
          flush
            ? "relative flex min-h-0 flex-1 flex-col"
            : "relative min-h-0 flex-1 overflow-auto p-3"
        }
      >
        {children}
      </div>
    </section>
  );
}
