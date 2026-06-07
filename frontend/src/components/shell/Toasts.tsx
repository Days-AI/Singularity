import { useEffect } from "react";
import { useSessionStore } from "@/store/sessionStore";

/** Transient notifications, primarily surfacing `error` SSE events. */
export function Toasts() {
  const toasts = useSessionStore((s) => s.toasts);
  const dismiss = useSessionStore((s) => s.dismissToast);

  return (
    <div className="pointer-events-none fixed bottom-12 right-4 z-[60] flex flex-col gap-2">
      {toasts.map((t) => (
        <ToastItem
          key={t.id}
          id={t.id}
          kind={t.kind}
          message={t.message}
          onDismiss={dismiss}
        />
      ))}
    </div>
  );
}

function ToastItem({
  id,
  kind,
  message,
  onDismiss,
}: {
  id: string;
  kind: "error" | "info";
  message: string;
  onDismiss: (id: string) => void;
}) {
  useEffect(() => {
    const timer = setTimeout(() => onDismiss(id), 6000);
    return () => clearTimeout(timer);
  }, [id, onDismiss]);

  const tone =
    kind === "error"
      ? "border-alert/60 text-alert"
      : "border-teal/60 text-teal";

  return (
    <div
      className={`pointer-events-auto flex max-w-sm items-start gap-2 rounded-sm border ${tone} bg-panel-raised px-3 py-2 shadow-panel animate-panel-in`}
    >
      <span className="mt-0.5 font-mono text-2xs uppercase tracking-widest">
        {kind === "error" ? "FAULT" : "INFO"}
      </span>
      <span className="font-mono text-xs text-data">{message}</span>
      <button
        onClick={() => onDismiss(id)}
        className="ml-auto font-mono text-xs text-muted hover:text-data"
      >
        x
      </button>
    </div>
  );
}
