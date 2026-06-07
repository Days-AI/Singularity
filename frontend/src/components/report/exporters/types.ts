import type { CapturedBlock } from "./capture";

export interface ReportExportInput {
  title: string;
  query: string;
  generatedAt: string;
  narrative: { title: string; content: string }[];
  blocks: CapturedBlock[];
  disclaimer: string;
}

export interface ParsedLine {
  text: string;
  bullet: boolean;
}

/** Strip lightweight markdown (**bold**, [VERIFY]) and classify bullet lines. */
export function parseNarrative(content: string): ParsedLine[] {
  const out: ParsedLine[] = [];
  for (const raw of content.split("\n")) {
    const line = raw.trim();
    if (!line) continue;
    const clean = line.replace(/\*\*(.*?)\*\*/g, "$1").replace(/\[VERIFY\]/g, "(verify)");
    if (clean.startsWith("- ")) out.push({ text: clean.slice(2).trim(), bullet: true });
    else out.push({ text: clean, bullet: false });
  }
  return out;
}

export function dataUrlToUint8(dataUrl: string): Uint8Array {
  const base64 = dataUrl.split(",")[1] ?? "";
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

export function formatTimestamp(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function triggerDownload(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
