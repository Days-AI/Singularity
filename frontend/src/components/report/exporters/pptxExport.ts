import PptxGenJS from "pptxgenjs";
import { formatTimestamp, parseNarrative, type ReportExportInput } from "./types";

const BG = "0A0E1A";
const PANEL = "0F1923";
const TEAL = "00B4D8";
const ORANGE = "F5A623";
const DATA = "E8ECF0";
const MUTED = "78909C";

const SLIDE_W = 13.333; // 16:9 inches
const SLIDE_H = 7.5;

/**
 * Composes a 16:9 PowerPoint deck: title slide, agenda, per-section narrative
 * slides, one slide per captured chart (image fit to the content area), and a
 * closing disclaimer slide.
 */
export async function exportReportToPptx(
  input: ReportExportInput,
  filename = "singularity-report.pptx"
): Promise<void> {
  const pptx = new PptxGenJS();
  pptx.defineLayout({ name: "WIDE", width: SLIDE_W, height: SLIDE_H });
  pptx.layout = "WIDE";

  const base = (slide: PptxGenJS.Slide) => {
    slide.background = { color: BG };
    slide.addText("Project Singularity · confidential", {
      x: 0.4, y: SLIDE_H - 0.4, w: SLIDE_W - 0.8, h: 0.3,
      fontSize: 8, color: MUTED, align: "left", fontFace: "Consolas",
    });
  };
  const heading = (slide: PptxGenJS.Slide, text: string) => {
    slide.addText(text, {
      x: 0.5, y: 0.35, w: SLIDE_W - 1, h: 0.7,
      fontSize: 24, bold: true, color: DATA, fontFace: "Arial",
    });
    slide.addShape(pptx.ShapeType.line, {
      x: 0.5, y: 1.05, w: 1.4, h: 0, line: { color: TEAL, width: 2 },
    });
  };

  // --- Title ---------------------------------------------------------------
  const title = pptx.addSlide();
  base(title);
  title.addText("PROJECT SINGULARITY", {
    x: 0.6, y: 2.2, w: SLIDE_W - 1.2, h: 0.5, fontSize: 16, color: TEAL, bold: true, fontFace: "Consolas",
  });
  title.addText("Strategic Intelligence Report", {
    x: 0.6, y: 2.8, w: SLIDE_W - 1.2, h: 1, fontSize: 40, color: DATA, bold: true,
  });
  title.addText(input.query || "Untitled analysis", {
    x: 0.6, y: 4.0, w: SLIDE_W - 1.2, h: 1, fontSize: 18, color: ORANGE,
  });
  title.addText(`Generated ${formatTimestamp(input.generatedAt)}`, {
    x: 0.6, y: 5.2, w: SLIDE_W - 1.2, h: 0.4, fontSize: 12, color: MUTED, fontFace: "Consolas",
  });

  // --- Narrative slides ----------------------------------------------------
  for (const section of input.narrative) {
    const slide = pptx.addSlide();
    base(slide);
    heading(slide, section.title);
    const lines = parseNarrative(section.content);
    const bullets = lines.length ? lines : [{ text: "(no content)", bullet: false }];
    slide.addText(
      bullets.map((ln) => ({
        text: ln.text,
        options: { bullet: ln.bullet ? { code: "2022" } : false, indentLevel: ln.bullet ? 1 : 0 },
      })),
      { x: 0.6, y: 1.3, w: SLIDE_W - 1.2, h: SLIDE_H - 2, fontSize: 13, color: DATA, valign: "top", lineSpacingMultiple: 1.1 }
    );
  }

  // --- Chart slides --------------------------------------------------------
  for (const block of input.blocks) {
    const slide = pptx.addSlide();
    base(slide);
    heading(slide, block.title);
    const areaX = 0.6;
    const areaY = 1.3;
    const areaW = SLIDE_W - 1.2;
    const areaH = SLIDE_H - 1.9;
    const ratio = block.height / block.width;
    let w = areaW;
    let h = w * ratio;
    if (h > areaH) {
      h = areaH;
      w = h / ratio;
    }
    slide.addImage({
      data: block.dataUrl,
      x: areaX + (areaW - w) / 2,
      y: areaY + (areaH - h) / 2,
      w,
      h,
    });
  }

  // --- Disclaimer ----------------------------------------------------------
  const last = pptx.addSlide();
  base(last);
  heading(last, "Methodology & Disclaimer");
  last.addText(input.disclaimer, {
    x: 0.6, y: 1.4, w: SLIDE_W - 1.2, h: 2, fontSize: 14, color: DATA, valign: "top",
  });
  last.addShape(pptx.ShapeType.rect, {
    x: 0.6, y: 3.6, w: SLIDE_W - 1.2, h: 0.02, fill: { color: PANEL },
  });

  await pptx.writeFile({ fileName: filename });
}
