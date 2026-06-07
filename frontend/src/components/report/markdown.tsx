import { Fragment, type ReactNode } from "react";

/**
 * Minimal markdown renderer shared by the inline ReportViewer and the expanded
 * ReportDocument: bullet lists, paragraphs, **bold**, and [VERIFY] badges.
 */
export function renderMarkdown(md: string): ReactNode {
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

export function renderInline(text: string): ReactNode {
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
