import {
  AlignmentType,
  Document,
  HeadingLevel,
  ImageRun,
  Packer,
  Paragraph,
  TextRun,
} from "docx";
import {
  dataUrlToUint8,
  formatTimestamp,
  parseNarrative,
  triggerDownload,
  type ReportExportInput,
} from "./types";

const MAX_IMG_W = 600;

/**
 * Composes a structured Word (.docx) report: title block, narrative sections as
 * headings/paragraphs/bullets, and one captioned figure per captured chart.
 */
export async function exportReportToWord(
  input: ReportExportInput,
  filename = "singularity-report.docx"
): Promise<void> {
  const children: Paragraph[] = [
    new Paragraph({ text: "Project Singularity", heading: HeadingLevel.TITLE }),
    new Paragraph({
      heading: HeadingLevel.HEADING_1,
      children: [new TextRun({ text: "Strategic Intelligence Report" })],
    }),
    new Paragraph({ children: [new TextRun({ text: input.query, italics: true, bold: true })] }),
    new Paragraph({
      children: [new TextRun({ text: `Generated ${formatTimestamp(input.generatedAt)}`, size: 18 })],
    }),
    new Paragraph({
      children: [new TextRun({ text: input.disclaimer, italics: true, size: 16, color: "888888" })],
    }),
  ];

  if (input.narrative.length) {
    children.push(new Paragraph({ text: "Narrative Analysis", heading: HeadingLevel.HEADING_1 }));
    for (const section of input.narrative) {
      children.push(new Paragraph({ text: section.title, heading: HeadingLevel.HEADING_2 }));
      for (const ln of parseNarrative(section.content)) {
        children.push(
          ln.bullet
            ? new Paragraph({ text: ln.text, bullet: { level: 0 } })
            : new Paragraph({ children: [new TextRun(ln.text)] })
        );
      }
    }
  }

  if (input.blocks.length) {
    children.push(new Paragraph({ text: "Visual Analytics", heading: HeadingLevel.HEADING_1 }));
    for (const block of input.blocks) {
      const width = Math.min(MAX_IMG_W, block.width);
      const height = (block.height / block.width) * width;
      children.push(new Paragraph({ text: block.title, heading: HeadingLevel.HEADING_3 }));
      try {
        children.push(
          new Paragraph({
            alignment: AlignmentType.CENTER,
            children: [
              new ImageRun({
                type: "png",
                data: dataUrlToUint8(block.dataUrl),
                transformation: { width, height },
              }),
            ],
          })
        );
      } catch {
        // skip unreadable image
      }
    }
  }

  const doc = new Document({ sections: [{ children }] });
  const blob = await Packer.toBlob(doc);
  triggerDownload(blob, filename);
}
