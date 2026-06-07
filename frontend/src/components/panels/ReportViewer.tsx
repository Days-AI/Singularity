import { useState } from "react";
import { createPortal } from "react-dom";
import { useSessionStore, type ReportSection } from "@/store/sessionStore";
import { renderMarkdown } from "@/components/report/markdown";
import { EnterpriseReport } from "@/components/report/EnterpriseReport";

export function ReportViewer() {
  const sections = useSessionStore((s) => s.reportSections);
  const connection = useSessionStore((s) => s.connection);
  const [open, setOpen] = useState(false);

  const canOpen = sections.length > 0;

  return (
    <div className="grid h-full min-h-0 grid-rows-[auto_minmax(0,1fr)] gap-1 overflow-hidden p-1">
      <div className="flex shrink-0 flex-wrap items-center justify-between gap-2">
        <span className="font-mono text-[10px] uppercase tracking-wider text-muted">
          {connection === "streaming"
            ? "synthesizing report..."
            : sections.length
              ? "report ready"
              : connection === "complete"
                ? "run resolved"
                : "awaiting run completion"}
        </span>
        <div className="panel-no-drag flex flex-wrap items-center gap-1.5">
          <button
            type="button"
            onClick={() => setOpen(true)}
            disabled={!canOpen}
            title={
              canOpen
                ? "Open the full enterprise report (charts, frameworks, PDF/Word/PPTX export)"
                : "Available once report sections are loaded"
            }
            className="rounded-sm border border-teal/50 bg-teal/10 px-2.5 py-1 font-mono text-2xs font-semibold uppercase tracking-wider text-teal transition-colors hover:bg-teal/20 disabled:opacity-40"
          >
            Open Full Report
          </button>
        </div>
      </div>

      {!sections.length ? (
        <div className="flex min-h-0 items-center justify-center font-mono text-xs text-muted">
          report synthesis pending
        </div>
      ) : (
        <div className="flex min-h-0 flex-col gap-3 overflow-auto pr-0.5">
          {sections.map((s) => (
            <SectionBlock key={s.index} section={s} />
          ))}
          {connection === "streaming" && (
            <div className="flex items-center gap-2 py-1 font-mono text-2xs text-teal">
              <span className="h-1.5 w-1.5 rounded-full bg-teal animate-pulse-stream" />
              synthesizing...
            </div>
          )}
        </div>
      )}

      {open && createPortal(<EnterpriseReport onClose={() => setOpen(false)} />, document.body)}
    </div>
  );
}

function SectionBlock({ section }: { section: ReportSection }) {
  return (
    <article className="animate-panel-in">
      <h3 className="mb-1.5 border-l-2 border-teal pl-2 font-display text-sm font-bold uppercase tracking-wider text-data">
        {section.title}
      </h3>
      <div className="space-y-1.5">{renderMarkdown(section.content)}</div>
    </article>
  );
}
