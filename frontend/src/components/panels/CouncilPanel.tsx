import { useSessionStore } from "@/store/sessionStore";
import { COLORS } from "@/lib/theme";
import type { CouncilOpinionPayload } from "@/types/events";

const ROLE_COLORS: Record<string, string> = {
  pr: COLORS.teal,
  brand: COLORS.orange,
  marketing: COLORS.data,
  consumer_psychology: COLORS.positive,
};

function SpecialistCard({ opinion }: { opinion: CouncilOpinionPayload }) {
  const accent = ROLE_COLORS[opinion.specialist_id] ?? COLORS.muted;
  return (
    <article
      className="rounded-sm border border-[color:var(--hairline)] bg-bg/40 p-2"
      style={{ borderLeftColor: accent, borderLeftWidth: 2 }}
    >
      <header className="mb-1 flex items-center justify-between gap-2">
        <span className="font-mono text-2xs uppercase tracking-wider" style={{ color: accent }}>
          {opinion.role}
        </span>
        <span className="font-mono text-2xs text-muted">
          {(opinion.confidence * 100).toFixed(0)}% conf
        </span>
      </header>
      <p className="line-clamp-3 font-mono text-2xs leading-relaxed text-data">
        {opinion.recommendation}
      </p>
      {opinion.risks.length > 0 && (
        <ul className="mt-1 space-y-0.5 font-mono text-2xs text-orange">
          {opinion.risks.slice(0, 2).map((r) => (
            <li key={r} className="truncate">
              · {r}
            </li>
          ))}
        </ul>
      )}
    </article>
  );
}

/** Four specialist cards plus optional council synthesis. */
export function CouncilPanel() {
  const opinions = useSessionStore((s) => s.councilOpinions);
  const council = useSessionStore((s) => s.council);

  const display = council?.opinions.length ? council.opinions : opinions;
  const synthesis = council?.synthesis ?? "";

  if (display.length === 0) {
    return (
      <div className="flex h-full items-center justify-center font-mono text-xs text-muted">
        awaiting specialist council
      </div>
    );
  }

  return (
    <div className="grid h-full min-h-0 grid-rows-[auto_minmax(0,1fr)] gap-1 overflow-y-auto p-1">
      <div className="grid shrink-0 grid-cols-2 gap-1">
        {display.map((op) => (
          <SpecialistCard key={op.specialist_id} opinion={op} />
        ))}
      </div>
      {synthesis && (
        <div className="shrink-0 rounded-sm border border-teal/30 bg-teal/5 p-2">
          <p className="mb-1 font-mono text-2xs uppercase tracking-wider text-teal">Synthesis</p>
          <p className="font-mono text-2xs leading-relaxed text-data">{synthesis}</p>
        </div>
      )}
    </div>
  );
}
