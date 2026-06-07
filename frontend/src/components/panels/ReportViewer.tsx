import { Fragment, type ReactNode } from "react";
import { useSessionStore, type ReportSection } from "@/store/sessionStore";

interface ReportViewerProps {
  onGenerateReport: () => void;
  reportLoading: boolean;
}

/**
 * Live-streaming McKinsey report viewer. Sections arrive via report_section
 * events and render through a minimal, dependency-free markdown subset
 * (headings via section title, **bold**, "- " bullets, and the [VERIFY] flag).
 *
 * Once a run is complete the user can regenerate the report on demand (e.g.
 * after adding focus questions) via the Generate Report action.
 */
export function ReportViewer({ onGenerateReport, reportLoading }: ReportViewerProps) {
  const sections = useSessionStore((s) => s.reportSections);
  const connection = useSessionStore((s) => s.connection);

  const canGenerate = connection === "complete" && !reportLoading;

  return (
    <div className="flex h-full flex-col gap-3">
      <div className="flex shrink-0 items-center justify-between">
        <span className="font-mono text-2xs uppercase tracking-widest text-muted">
          {reportLoading
            ? "synthesizing report..."
            : connection === "complete"
              ? "run resolved"
              : "awaiting run completion"}
        </span>
        <button
          onClick={onGenerateReport}
          disabled={!canGenerate}
          title={
            connection === "complete"
              ? "Regenerate the strategic report"
              : "Available after the simulation completes"
          }
          className="rounded-sm border border-teal/50 bg-teal/10 px-2.5 py-1 font-mono text-2xs font-semibold uppercase tracking-wider text-teal transition-colors hover:bg-teal/20 disabled:opacity-40"
        >
          {reportLoading ? "Generating..." : "Generate Report"}
        </button>
      </div>

      {!sections.length ? (
        <div className="flex flex-1 items-center justify-center font-mono text-xs text-muted">
          report synthesis pending
        </div>
      ) : (
        <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-auto pr-1">
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

function renderMarkdown(md: string): ReactNode {
  const lines = md.split("\n");
  const blocks: ReactNode[] = [];
  let bullets: ReactNode[] = [];

  const flush = () => {
    if (bullets.length) {
      blocks.push(
        <ul key={`ul-${blocks.length}`} className="ml-1 space-y-1">
          {bullets}
        </ul>
      );
      bullets = [];
    }
  };

  lines.forEach((raw, i) => {
    const line = raw.trim();
    if (!line) {
      flush();
      return;
    }
    if (line.startsWith("- ")) {
      bullets.push(
        <li key={`li-${i}`} className="flex gap-2 font-mono text-xs leading-relaxed text-data">
          <span className="text-teal">-</span>
          <span>{renderInline(line.slice(2))}</span>
        </li>
      );
    } else {
      flush();
      blocks.push(
        <p key={`p-${i}`} className="font-mono text-xs leading-relaxed text-data/90">
          {renderInline(line)}
        </p>
      );
    }
  });
  flush();
  return blocks;
}

/** Inline: **bold** and [VERIFY] flag highlighting. */
function renderInline(text: string): ReactNode {
  const parts = text.split(/(\*\*[^*]+\*\*|\[VERIFY\])/g);
  return parts.map((part, i) => {
    if (part === "[VERIFY]") {
      return (
        <span
          key={i}
          className="mx-0.5 rounded-sm bg-alert/15 px-1 font-semibold text-alert"
        >
          [VERIFY]
        </span>
      );
    }
    if (part.startsWith("**") && part.endsWith("**")) {
      return (
        <strong key={i} className="font-semibold text-teal">
          {part.slice(2, -2)}
        </strong>
      );
    }
    return <Fragment key={i}>{part}</Fragment>;
  });
}
