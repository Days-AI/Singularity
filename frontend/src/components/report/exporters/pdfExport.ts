import { jsPDF } from "jspdf";
import { formatTimestamp, parseNarrative, type ReportExportInput } from "./types";

/** Dark theme palette mirrored to RGB tuples for jsPDF. */
const RGB = {
  bg: [10, 14, 26] as const,
  panel: [15, 25, 35] as const,
  teal: [0, 180, 216] as const,
  orange: [245, 166, 35] as const,
  data: [232, 236, 240] as const,
  muted: [120, 144, 156] as const,
};

const MARGIN = 40;
const A4_W = 595.28;
const A4_H = 841.89;
const CONTENT_W = A4_W - MARGIN * 2;

/**
 * Composes a branded, multi-section PDF: cover page, narrative sections, and
 * one figure per captured visualization. Charts arrive as pre-rendered dark
 * PNGs (see capture.ts), so they stay readable on the dark pages.
 */
export async function exportReportToPdf(
  input: ReportExportInput,
  filename = "singularity-report.pdf"
): Promise<void> {
  const pdf = new jsPDF({ orientation: "portrait", unit: "pt", format: "a4" });
  let y = MARGIN;
  let page = 1;

  const paintBackground = () => {
    pdf.setFillColor(...RGB.bg);
    pdf.rect(0, 0, A4_W, A4_H, "F");
  };
  const footer = () => {
    pdf.setFontSize(7);
    pdf.setTextColor(...RGB.muted);
    pdf.text("Project Singularity · confidential", MARGIN, A4_H - 18);
    pdf.text(`${page}`, A4_W - MARGIN, A4_H - 18, { align: "right" });
  };
  const newPage = () => {
    footer();
    pdf.addPage();
    page += 1;
    paintBackground();
    y = MARGIN;
  };
  const ensure = (needed: number) => {
    if (y + needed > A4_H - MARGIN) newPage();
  };

  paintBackground();

  // --- Cover ---------------------------------------------------------------
  y = 150;
  pdf.setFont("helvetica", "bold");
  pdf.setFontSize(13);
  pdf.setTextColor(...RGB.teal);
  pdf.text("PROJECT SINGULARITY", MARGIN, y);
  y += 34;
  pdf.setFontSize(26);
  pdf.setTextColor(...RGB.data);
  pdf.text("Strategic Intelligence Report", MARGIN, y, { maxWidth: CONTENT_W });
  y += 40;
  pdf.setFont("helvetica", "normal");
  pdf.setFontSize(12);
  pdf.setTextColor(...RGB.orange);
  for (const line of pdf.splitTextToSize(input.query || "Untitled analysis", CONTENT_W)) {
    pdf.text(line, MARGIN, y);
    y += 18;
  }
  y += 14;
  pdf.setFontSize(9);
  pdf.setTextColor(...RGB.muted);
  pdf.text(`Generated ${formatTimestamp(input.generatedAt)}`, MARGIN, y);
  y += 30;
  pdf.setFontSize(8);
  for (const line of pdf.splitTextToSize(input.disclaimer, CONTENT_W)) {
    pdf.text(line, MARGIN, y);
    y += 12;
  }
  newPage();

  // --- Narrative sections --------------------------------------------------
  if (input.narrative.length) {
    sectionHeading(pdf, "Narrative Analysis", y);
    y += 26;
    for (const section of input.narrative) {
      ensure(40);
      pdf.setFont("helvetica", "bold");
      pdf.setFontSize(12);
      pdf.setTextColor(...RGB.teal);
      pdf.text(section.title, MARGIN, y);
      y += 18;
      pdf.setFont("helvetica", "normal");
      pdf.setFontSize(9.5);
      pdf.setTextColor(...RGB.data);
      for (const ln of parseNarrative(section.content)) {
        const indent = ln.bullet ? 14 : 0;
        const prefix = ln.bullet ? "•  " : "";
        const lines = pdf.splitTextToSize(prefix + ln.text, CONTENT_W - indent);
        for (const l of lines) {
          ensure(14);
          pdf.text(l, MARGIN + indent, y);
          y += 13;
        }
      }
      y += 10;
    }
    newPage();
  }

  // --- Visualizations ------------------------------------------------------
  sectionHeading(pdf, "Visual Analytics", y);
  y += 26;
  for (const block of input.blocks) {
    const imgW = CONTENT_W;
    const imgH = Math.min((block.height / block.width) * imgW, A4_H - MARGIN * 2 - 30);
    ensure(imgH + 28);
    pdf.setFont("helvetica", "bold");
    pdf.setFontSize(10);
    pdf.setTextColor(...RGB.orange);
    pdf.text(block.title, MARGIN, y);
    y += 12;
    try {
      pdf.addImage(block.dataUrl, "PNG", MARGIN, y, imgW, imgH, undefined, "FAST");
    } catch {
      // skip unreadable image
    }
    y += imgH + 18;
  }

  footer();
  pdf.save(filename);
}

function sectionHeading(pdf: jsPDF, text: string, y: number) {
  pdf.setFont("helvetica", "bold");
  pdf.setFontSize(15);
  pdf.setTextColor(...RGB.data);
  pdf.text(text, MARGIN, y);
  pdf.setDrawColor(0, 180, 216);
  pdf.setLineWidth(1.2);
  pdf.line(MARGIN, y + 6, MARGIN + 60, y + 6);
}
