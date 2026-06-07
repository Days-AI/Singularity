import type { ReactNode } from "react";

/**
 * A titled card that marks its content as an exportable block. The export
 * pipeline (`capture.ts`) finds every `[data-capture]` node, reads its
 * `data-capture-title`, and rasterizes it (Plotly.toImage for charts,
 * html2canvas for plain-DOM framework blocks).
 */
export function CaptureCard({
  title,
  subtitle,
  height = 300,
  span = 1,
  children,
}: {
  title: string;
  subtitle?: string;
  height?: number;
  span?: 1 | 2;
  children: ReactNode;
}) {
  return (
    <section
      data-capture
      data-capture-title={title}
      className={`rounded-md border border-[color:var(--hairline)] bg-panel/60 p-3 ${
        span === 2 ? "md:col-span-2" : ""
      }`}
    >
      <div className="mb-2 flex items-baseline justify-between gap-2">
        <h4 className="font-mono text-2xs uppercase tracking-widest text-muted">{title}</h4>
        {subtitle && <span className="font-mono text-2xs text-muted/70">{subtitle}</span>}
      </div>
      <div style={{ height }}>{children}</div>
    </section>
  );
}

/** Auto-height capturable wrapper for plain-DOM framework blocks (KPI, SWOT...). */
export function CaptureBlock({
  title,
  children,
}: {
  title: string;
  children: ReactNode;
}) {
  return (
    <section data-capture data-capture-title={title} className="space-y-3">
      {children}
    </section>
  );
}

export function ChartEmpty({ label }: { label: string }) {
  return (
    <div className="flex h-full items-center justify-center font-mono text-xs text-muted">
      {label}
    </div>
  );
}
