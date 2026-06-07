import type { SSEvent } from "@/types/events";

/**
 * Coalesces high-frequency SSE events onto animation frames so persona_batch
 * floods do not synchronously re-render the entire dashboard 6+ times per second.
 */
export function createBatchedDispatch(onEvent: (event: SSEvent) => void) {
  let queue: SSEvent[] = [];
  let raf = 0;

  const flush = () => {
    raf = 0;
    if (!queue.length) return;

    // Keep only the latest persona_batch per frame; apply others in order.
    const pending: SSEvent[] = [];
    let latestPersona: SSEvent | null = null;

    for (const ev of queue) {
      if (ev.type === "persona_batch") {
        latestPersona = ev;
      } else {
        pending.push(ev);
      }
    }
    queue = [];

    for (const ev of pending) onEvent(ev);
    if (latestPersona) onEvent(latestPersona);
  };

  return (event: SSEvent) => {
    queue.push(event);
    if (!raf) {
      raf = requestAnimationFrame(flush);
    }
  };
}
