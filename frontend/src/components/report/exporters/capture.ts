import html2canvas from "html2canvas";
import Plotly from "plotly.js-dist-min";
import { COLORS } from "@/lib/theme";

export interface CapturedBlock {
  title: string;
  dataUrl: string;
  width: number;
  height: number;
}

/**
 * Composite a (possibly transparent) chart PNG onto a solid dark card so the
 * light-themed axes/labels stay readable when embedded in light documents.
 */
function compositeOnDark(dataUrl: string, width: number, height: number): Promise<string> {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = img.width || width;
      canvas.height = img.height || height;
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        resolve(dataUrl);
        return;
      }
      ctx.fillStyle = COLORS.bg;
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0);
      resolve(canvas.toDataURL("image/png"));
    };
    img.onerror = () => resolve(dataUrl);
    img.src = dataUrl;
  });
}

/**
 * Rasterize every `[data-capture]` block inside `container` to a PNG data URL.
 *
 * Plotly charts are exported via `Plotly.toImage` (reliable for SVG, WebGL/gl3d,
 * and polar traces - which html2canvas cannot capture). Plain-DOM framework
 * blocks (KPI cards, SWOT, Porter) fall back to html2canvas. Blocks are returned
 * in DOM order so the document composer can lay them out top-to-bottom.
 */
export async function captureReport(container: HTMLElement): Promise<CapturedBlock[]> {
  const nodes = Array.from(
    container.querySelectorAll<HTMLElement>("[data-capture]")
  );
  const blocks: CapturedBlock[] = [];

  for (const node of nodes) {
    const title = node.getAttribute("data-capture-title") ?? "";
    const plotEl = node.querySelector<HTMLElement>(".js-plotly-plot");
    try {
      if (plotEl) {
        const rect = plotEl.getBoundingClientRect();
        const width = Math.max(320, Math.round(rect.width));
        const height = Math.max(240, Math.round(rect.height));
        const raw: string = await Plotly.toImage(plotEl, {
          format: "png",
          width,
          height,
          scale: 2,
        });
        const dataUrl = await compositeOnDark(raw, width * 2, height * 2);
        blocks.push({ title, dataUrl, width, height });
      } else {
        const canvas = await html2canvas(node, {
          backgroundColor: COLORS.bg,
          scale: 2,
          useCORS: true,
          logging: false,
        });
        blocks.push({
          title,
          dataUrl: canvas.toDataURL("image/png"),
          width: canvas.width,
          height: canvas.height,
        });
      }
    } catch {
      // Best-effort: a single failed block must not abort the whole export.
    }
  }

  return blocks;
}
