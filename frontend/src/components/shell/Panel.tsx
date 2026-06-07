import { useCallback, useEffect, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";

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

function bodyClass(flush: boolean): string {
  return flush
    ? "relative flex min-h-0 flex-1 flex-col overflow-hidden"
    : "relative flex min-h-0 flex-1 flex-col overflow-hidden p-1";
}

function ExpandIcon() {
  return (
    <svg viewBox="0 0 16 16" width="12" height="12" fill="none" aria-hidden="true">
      <path
        d="M2 6V2h4M14 6V2h-4M2 10v4h4M14 10v4h-4"
        stroke="currentColor"
        strokeWidth="1.4"
        strokeLinecap="round"
      />
    </svg>
  );
}

function CloseIcon() {
  return (
    <svg viewBox="0 0 16 16" width="13" height="13" fill="none" aria-hidden="true">
      <path
        d="M3 3l10 10M13 3L3 13"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
      />
    </svg>
  );
}

export function Panel({
  label,
  code,
  status = "idle",
  order = 0,
  className = "",
  flush = false,
  children,
}: PanelProps) {
  const [expanded, setExpanded] = useState(false);

  const close = useCallback(() => setExpanded(false), []);

  useEffect(() => {
    if (!expanded) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") close();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [expanded, close]);

  const headerControls = (
    <div className="panel-no-drag flex items-center gap-2">
      {code && (
        <span className="font-mono text-2xs tracking-widest text-muted">{code}</span>
      )}
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        className="flex h-5 w-5 items-center justify-center rounded text-muted transition-colors hover:bg-[color:var(--hairline)] hover:text-teal"
        title={expanded ? "Exit full view (Esc)" : "Full view"}
        aria-label={expanded ? "Exit full view" : "Full view"}
      >
        {expanded ? <CloseIcon /> : <ExpandIcon />}
      </button>
    </div>
  );

  return (
    <section
      className={`group flex h-full min-h-0 flex-col rounded-[var(--panel-radius)] border border-[color:var(--hairline)] bg-panel shadow-panel animate-panel-in ${className}`}
      style={{ animationDelay: `${order * 70}ms` }}
    >
      <header className="panel-drag-handle flex cursor-grab items-center justify-between border-b border-[color:var(--hairline)] px-3 py-1.5 active:cursor-grabbing">
        <div className="flex min-w-0 items-center gap-2">
          <span className={`h-1.5 w-1.5 shrink-0 rounded-full ${STATUS_DOT[status]}`} />
          <h2 className="panel-label truncate">{label}</h2>
        </div>
        {headerControls}
      </header>

      {/* When expanded the single children instance lives in the modal; the tile
          shows a lightweight placeholder so heavy charts never double-mount. */}
      {expanded ? (
        <div className="flex min-h-0 flex-1 items-center justify-center p-3 text-center">
          <span className="font-mono text-2xs uppercase tracking-widest text-muted">
            Open in full view
          </span>
        </div>
      ) : (
        <div className={bodyClass(flush)}>{children}</div>
      )}

      {expanded &&
        createPortal(
          <div
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4 backdrop-blur-sm animate-panel-in"
            onMouseDown={(e) => {
              if (e.target === e.currentTarget) close();
            }}
          >
            <section className="flex h-[90vh] w-[92vw] max-w-[1600px] flex-col overflow-hidden rounded-[var(--panel-radius)] border border-[color:var(--hairline)] bg-panel shadow-panel">
              <header className="flex items-center justify-between border-b border-[color:var(--hairline)] px-4 py-2">
                <div className="flex items-center gap-2">
                  <span className={`h-1.5 w-1.5 rounded-full ${STATUS_DOT[status]}`} />
                  <h2 className="panel-label">{label}</h2>
                </div>
                <div className="flex items-center gap-2">
                  {code && (
                    <span className="font-mono text-2xs tracking-widest text-muted">
                      {code}
                    </span>
                  )}
                  <button
                    type="button"
                    onClick={close}
                    className="flex h-6 w-6 items-center justify-center rounded text-muted transition-colors hover:bg-[color:var(--hairline)] hover:text-teal"
                    title="Exit full view (Esc)"
                    aria-label="Exit full view"
                  >
                    <CloseIcon />
                  </button>
                </div>
              </header>
              <div className={bodyClass(flush)}>{children}</div>
            </section>
          </div>,
          document.body
        )}
    </section>
  );
}
