"""Canonical Singularity application use-case catalog for report synthesis."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class UseCase:
    domain: str
    tagline: str
    description: str


USE_CASES: tuple[UseCase, ...] = (
    UseCase(
        domain="Communications & PR",
        tagline="Craft Narratives",
        description=(
            "Test different communication strategies to generate the desired audience response."
        ),
    ),
    UseCase(
        domain="Product",
        tagline="Decide Features",
        description=(
            "Evaluate how target customers react to product concepts, ideas, and new features."
        ),
    ),
    UseCase(
        domain="Branding",
        tagline="Stand Out",
        description=(
            "Test how different brand identities, messaging, and voice options resonate "
            "with your ideal customers."
        ),
    ),
    UseCase(
        domain="Social Media",
        tagline="Create Content",
        description=(
            "Test social media content within simulations of your audience and network."
        ),
    ),
    UseCase(
        domain="Marketing",
        tagline="Generate Leads",
        description=(
            "Evaluate marketing campaigns and content using simulations of your target customers."
        ),
    ),
    UseCase(
        domain="Public Policy",
        tagline="Capture Attention",
        description=(
            "Test policies, initiatives, and public messaging to maximize public acceptance "
            "and engagement."
        ),
    ),
)

DOMAIN_ORDER = [uc.domain for uc in USE_CASES]


def use_cases_as_dicts() -> list[dict[str, str]]:
    return [
        {"domain": uc.domain, "tagline": uc.tagline, "description": uc.description}
        for uc in USE_CASES
    ]


def _cluster_summary(data: dict[str, Any]) -> list[dict[str, Any]]:
    delib = data.get("deliberation") or {}
    clusters = delib.get("narrative_clusters") or []
    if clusters:
        return clusters
    archetypes = delib.get("persona_archetypes") or []
    return [
        {
            "label": a.get("name", "Segment"),
            "share": (a.get("size", 0) / max(data.get("population", 1), 1)),
            "stance": "mixed",
            "key_themes": (a.get("concerns") or [])[:2],
        }
        for a in archetypes[:3]
    ]


def _top_themes(data: dict[str, Any]) -> list[str]:
    clusters = _cluster_summary(data)
    themes: list[str] = []
    for c in clusters:
        for t in c.get("key_themes") or []:
            if t and t not in themes:
                themes.append(str(t))
    return themes[:4]


def _decision_snippet(data: dict[str, Any], index: int = 0) -> str | None:
    options = data.get("decision_engine") or []
    if not options or index >= len(options):
        return None
    opt = options[index]
    action = str(opt.get("action", "")).strip()
    if not action:
        return None
    return action[:160]


def insight_for_domain(domain: str, data: dict[str, Any]) -> str:
    """Deterministic simulation-grounded insight per application domain."""
    sent = data.get("mean_sentiment")
    population = data.get("population", 1500)
    delib = data.get("deliberation") or {}
    agreement = delib.get("agreement_rate")
    polarization = delib.get("polarization_index")
    clusters = _cluster_summary(data)
    themes = _top_themes(data)
    theme_txt = ", ".join(themes[:2]) if themes else "value perception and trust"
    sent_txt = f"{sent:+.2f}" if sent is not None else "mixed"
    cluster_bits = []
    for c in clusters[:3]:
        label = c.get("label", "Segment")
        share = c.get("share", 0)
        pct = int(round(float(share) * 100)) if share else 0
        cluster_bits.append(f"{label} ({pct}%)")
    cluster_txt = "; ".join(cluster_bits) if cluster_bits else "segment mix unavailable"

    if domain == "Communications & PR":
        base = (
            f"Mean population sentiment {sent_txt} across {population:,} agents; "
            f"clusters: {cluster_txt}. Test narrative variants against dominant themes: {theme_txt}."
        )
        dec = _decision_snippet(data, 0)
        return f"{base} Top-ranked action: {dec}." if dec else base

    if domain == "Product":
        enthusiast = next((c for c in clusters if "enthus" in str(c.get("label", "")).lower()), None)
        skeptic = next((c for c in clusters if "skept" in str(c.get("label", "")).lower()), None)
        parts = [f"Aggregate sentiment {sent_txt} signals concept reception."]
        if enthusiast:
            parts.append(
                f"Enthusiast share ~{int(float(enthusiast.get('share', 0)) * 100)}% — prioritize features they cite."
            )
        if skeptic:
            parts.append(
                f"Skeptic share ~{int(float(skeptic.get('share', 0)) * 100)}% — validate objections before launch."
            )
        return " ".join(parts)

    if domain == "Branding":
        om = data.get("ocean_mean") or {}
        o, a = om.get("O"), om.get("A")
        ocean_note = ""
        if o is not None and a is not None:
            ocean_note = f" Mean OCEAN Openness {o:.0f} / Agreeableness {a:.0f} — "
            ocean_note += "high-O audiences favor distinctive voice; high-A favor warm, inclusive tone."
        return (
            f"Segment divergence ({cluster_txt}) implies multiple brand voices may be needed.{ocean_note} "
            f"Simulate identity options against {population:,} profiles before committing."
        ).strip()

    if domain == "Social Media":
        return (
            f"Network simulation shows {cluster_txt} with sentiment {sent_txt}. "
            f"Draft content against themes: {theme_txt}; A/B test posts in the synthetic audience feed."
        )

    if domain == "Marketing":
        mc = data.get("monte_carlo") or {}
        p50 = (mc.get("outcome_percentiles") or {}).get("p50")
        forecast = data.get("forecast") or {}
        fc_note = ""
        if forecast.get("pct_change") is not None:
            fc_note = f" Forecast {forecast.get('metric', 'index')} {forecast['pct_change']:+.1f}% over {forecast.get('horizon_days', 90)}d."
        p50_note = f" Monte Carlo median outcome {p50:.0f}/100." if p50 is not None else ""
        dec = _decision_snippet(data, 1)
        lead = (
            f"Campaign simulation: {cluster_txt}; mean sentiment {sent_txt}.{fc_note}{p50_note} "
            f"Route spend toward highest-action clusters first."
        )
        return f"{lead} Alternative path: {dec}." if dec else lead

    if domain == "Public Policy":
        agr = f"{float(agreement):.0%}" if agreement is not None else "unknown"
        pol = f"{float(polarization):.2f}" if polarization is not None else "unknown"
        causal = data.get("causal_drivers") or []
        policy_edge = next(
            (d for d in causal if "policy" in str(d.get("cause", "")).lower()
             or "policy" in str(d.get("effect", "")).lower()),
            None,
        )
        edge_note = ""
        if policy_edge:
            edge_note = (
                f" Causal link {policy_edge['cause']} → {policy_edge['effect']} "
                f"(p={policy_edge.get('p_value')})."
            )
        return (
            f"Deliberation agreement {agr}, polarization index {pol}.{edge_note} "
            f"Test messaging variants to move undecided segments before rollout."
        )

    return f"Population sentiment {sent_txt}; simulate interventions across {population:,} agents."


def build_application_playbooks_markdown(
    data: dict[str, Any],
    llm_recommendations: dict[str, str] | None = None,
) -> str:
    """Render the Simulation Applications section as markdown."""
    llm_recommendations = llm_recommendations or {}
    blocks: list[str] = []
    for uc in USE_CASES:
        insight = insight_for_domain(uc.domain, data)
        llm_extra = llm_recommendations.get(uc.domain, "").strip()
        block = (
            f"### {uc.domain} — {uc.tagline}\n"
            f"{uc.description}\n\n"
            f"- **Simulation insight:** {insight}"
        )
        if llm_extra:
            block += f"\n- **Recommended action:** {llm_extra}"
        blocks.append(block)
    return "\n\n".join(blocks)


def merge_llm_playbooks(polished: dict[str, Any]) -> dict[str, str]:
    """Extract domain -> recommendation map from polish JSON."""
    raw = polished.get("application_playbooks")
    if not isinstance(raw, list):
        return {}
    out: dict[str, str] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        domain = str(item.get("domain", "")).strip()
        rec = str(item.get("recommendation", "")).strip()
        if domain and rec:
            out[domain] = rec
    return out
