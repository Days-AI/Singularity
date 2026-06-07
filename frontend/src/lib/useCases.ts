/** Canonical Singularity application use-case catalog (mirrors backend/report/use_cases.py). */

export interface UseCase {
  domain: string;
  tagline: string;
  description: string;
}

export const USE_CASES: UseCase[] = [
  {
    domain: "Communications & PR",
    tagline: "Craft Narratives",
    description:
      "Test different communication strategies to generate the desired audience response.",
  },
  {
    domain: "Product",
    tagline: "Decide Features",
    description:
      "Evaluate how target customers react to product concepts, ideas, and new features.",
  },
  {
    domain: "Branding",
    tagline: "Stand Out",
    description:
      "Test how different brand identities, messaging, and voice options resonate with your ideal customers.",
  },
  {
    domain: "Social Media",
    tagline: "Create Content",
    description:
      "Test social media content within simulations of your audience and network.",
  },
  {
    domain: "Marketing",
    tagline: "Generate Leads",
    description:
      "Evaluate marketing campaigns and content using simulations of your target customers.",
  },
  {
    domain: "Public Policy",
    tagline: "Capture Attention",
    description:
      "Test policies, initiatives, and public messaging to maximize public acceptance and engagement.",
  },
];

export interface ParsedPlaybook {
  domain: string;
  tagline: string;
  description: string;
  simulationInsight: string;
  recommendedAction?: string;
}

/** Parse the Simulation Applications markdown section into structured playbooks. */
export function parseApplicationSection(content: string): ParsedPlaybook[] {
  if (!content.trim()) return [];

  const chunks = content.split(/^### /m).filter(Boolean);
  const parsed: ParsedPlaybook[] = [];

  for (const chunk of chunks) {
    const lines = chunk.trim().split("\n");
    const header = lines[0]?.trim() ?? "";
    const dash = header.indexOf(" — ");
    const domain = dash >= 0 ? header.slice(0, dash).trim() : header;
    const tagline = dash >= 0 ? header.slice(dash + 3).trim() : "";

    const canonical = USE_CASES.find((u) => u.domain === domain);
    const bodyLines = lines.slice(1);
    const descLines: string[] = [];
    let simulationInsight = "";
    let recommendedAction: string | undefined;

    for (const line of bodyLines) {
      const insightMatch = line.match(/^\-\s+\*\*Simulation insight:\*\*\s*(.+)$/i);
      const actionMatch = line.match(/^\-\s+\*\*Recommended action:\*\*\s*(.+)$/i);
      if (insightMatch) {
        simulationInsight = insightMatch[1].trim();
      } else if (actionMatch) {
        recommendedAction = actionMatch[1].trim();
      } else if (!line.startsWith("-") && line.trim()) {
        descLines.push(line.trim());
      }
    }

    parsed.push({
      domain,
      tagline: tagline || canonical?.tagline || "",
      description: descLines.join(" ") || canonical?.description || "",
      simulationInsight,
      recommendedAction,
    });
  }

  if (parsed.length >= USE_CASES.length) return parsed;

  // Fallback: merge with canonical catalog when parsing yields partial results.
  return USE_CASES.map((uc) => {
    const found = parsed.find((p) => p.domain === uc.domain);
    return (
      found ?? {
        domain: uc.domain,
        tagline: uc.tagline,
        description: uc.description,
        simulationInsight: "",
      }
    );
  });
}
