"""Decision engine (spec stage 10).

Synthesizes five ranked strategic options from prediction market,
Monte Carlo, causal, swarm, and forecast signals.
"""
from __future__ import annotations

import logging
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from agents._async_util import run_optional_async
from config import get_settings
from llm.ollama_client import get_ollama
from prompts import DECISION_ENGINE_USER, SINGULARITY_ENGINE_SYSTEM
from state import SingularityState

logger = logging.getLogger("singularity.decision_engine")

_OPTION_TYPES = ["best", "alternative", "high_risk", "low_risk", "experimental"]


@dataclass
class DecisionOption:
    type: str
    action: str
    expected_outcome: str
    supporting_evidence: list[str] = field(default_factory=list)
    causal_drivers: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    confidence: float = 0.5


@dataclass
class DecisionEngineResult:
    options: list[DecisionOption] = field(default_factory=list)


def run(state: SingularityState) -> DecisionEngineResult:
    template_options = _build_template_options(state)
    if get_settings().decision_engine_llm_enrich:
        enriched = run_optional_async(_enrich_with_llm(state, template_options))
        if enriched:
            return enriched
    return DecisionEngineResult(options=template_options)


def to_metrics(result: DecisionEngineResult) -> dict[str, Any]:
    return {
        "options": [
            {
                "type": o.type,
                "action": o.action,
                "expected_outcome": o.expected_outcome,
                "supporting_evidence": o.supporting_evidence,
                "causal_drivers": o.causal_drivers,
                "risks": o.risks,
                "confidence": round(o.confidence, 3),
            }
            for o in result.options
        ],
    }


def _build_template_options(state: SingularityState) -> list[DecisionOption]:
    deliberation = state.metrics.get("deliberation", {})
    consensus = state.metrics.get("consensus", {})
    market = state.metrics.get("prediction_market", {})
    mc = state.metrics.get("monte_carlo", {})
    swarm = state.metrics.get("swarm_optimization", {})
    agreement = float(
        consensus.get("agreement_score", deliberation.get("agreement_rate", 0.5))
    )
    polarization = float(deliberation.get("polarization_index", 0.2))
    consensus_action = str(consensus.get("recommended_action", "")).strip()
    overall = float(market.get("overall_outcome", 50.0))
    p50 = float(mc.get("outcome_percentiles", {}).get("p50", overall))
    p95 = float(mc.get("outcome_percentiles", {}).get("p95", overall + 15))
    p5 = float(mc.get("outcome_percentiles", {}).get("p5", overall - 15))
    swarm_steps = swarm.get("optimal_path", {}).get("steps", [])
    swarm_domain = swarm.get("domain", "consumer_research")

    causal_drivers = _top_causal_drivers(state)
    evidence_bullets = _evidence_bullets(state)
    narrative_gaps = _narrative_gaps(deliberation)

    path_hint = " → ".join(str(s) for s in swarm_steps[:4]) if swarm_steps else "segment-targeted messaging sequence"
    best_action = (
        consensus_action[:400]
        if consensus_action
        else (
            f"Execute the swarm-optimized {swarm_domain} pathway: {path_hint}. "
            f"Prioritize segments showing convergence (agreement {agreement:.0%})."
        )
    )

    return [
        DecisionOption(
            type="best",
            action=best_action,
            expected_outcome=f"Median outcome {p50:.0f}/100 with upside toward {p95:.0f}/100",
            supporting_evidence=evidence_bullets[:3],
            causal_drivers=causal_drivers[:3],
            risks=["Moderate execution dependency on narrative consistency"],
            confidence=float(np.clip(agreement * 0.7 + (overall / 100) * 0.3, 0.3, 0.95)),
        ),
        DecisionOption(
            type="alternative",
            action=(
                "Deploy phased rollout: test messaging with Pragmatist cluster first, "
                "then expand to Enthusiasts while monitoring Skeptic response."
            ),
            expected_outcome=f"Steady progression toward {overall:.0f}/100 market aggregate",
            supporting_evidence=evidence_bullets[1:4] if len(evidence_bullets) > 1 else evidence_bullets,
            causal_drivers=causal_drivers[1:4] if len(causal_drivers) > 1 else causal_drivers,
            risks=["Slower time-to-impact vs full launch"],
            confidence=float(np.clip(0.55 + agreement * 0.25, 0.35, 0.85)),
        ),
        DecisionOption(
            type="high_risk",
            action=(
                "Aggressive bold-positioning campaign targeting viral amplification "
                f"and black-swan upside (tail scenario {p95:.0f}/100)."
            ),
            expected_outcome=f"High variance: {p5:.0f}–{p95:.0f}/100 outcome range",
            supporting_evidence=["Polarization index supports high-reward/high-risk bifurcation"],
            causal_drivers=causal_drivers[:2],
            risks=["Backlash amplification", "Trust erosion if claims exceed evidence", "Regulatory/reputation exposure"],
            confidence=float(np.clip(0.25 + (1 - agreement) * 0.3, 0.15, 0.55)),
        ),
        DecisionOption(
            type="low_risk",
            action=(
                "Conservative trust-building: evidence-first communications, "
                "transparent objection handling, minimal creative disruption."
            ),
            expected_outcome=f"Floor-protected outcome near {max(p5, p50 - 10):.0f}/100 with limited upside",
            supporting_evidence=evidence_bullets[:2],
            causal_drivers=["trust", "credibility_proof"] + causal_drivers[:1],
            risks=["Missed adoption window", "Competitor narrative capture"],
            confidence=float(np.clip(0.65 + agreement * 0.2, 0.5, 0.9)),
        ),
        DecisionOption(
            type="experimental",
            action=(
                f"Pilot micro-segment experiment addressing: {narrative_gaps}. "
                "A/B test swarm alternative paths before scale commitment."
            ),
            expected_outcome="Learning-oriented; outcome data reduces uncertainty by 15–25%",
            supporting_evidence=["Deliberation reveals unresolved narrative clusters"],
            causal_drivers=causal_drivers[:2],
            risks=["Pilot may not generalize", "Resource allocation to non-scaled test"],
            confidence=float(np.clip(0.4 + (1 - polarization) * 0.3, 0.3, 0.7)),
        ),
    ]


def _top_causal_drivers(state: SingularityState) -> list[str]:
    if not state.causal:
        return ["population sentiment", "audience trust"]
    node_label = {n.id: n.label for n in state.causal.nodes}
    drivers = []
    for e in sorted(state.causal.edges, key=lambda x: -x.weight)[:5]:
        cause = node_label.get(e.source, e.source)
        effect = node_label.get(e.target, e.target)
        drivers.append(f"{cause} → {effect}")
    return drivers or ["population sentiment"]


def _evidence_bullets(state: SingularityState) -> list[str]:
    bullets = []
    for e in state.evidence[:5]:
        bullets.append(f"{e.source}: {e.title[:80]}")
    if state.persona_responses:
        sent = float(np.mean([r.sentiment_score for r in state.persona_responses]))
        bullets.append(f"Population sentiment {sent:+.2f} across simulated personas")
    return bullets or ["Limited external evidence; rely on simulation signals"]


def _narrative_gaps(deliberation: dict) -> str:
    clusters = deliberation.get("narrative_clusters", [])
    opposed = [c["label"] for c in clusters if c.get("stance") == "opposed"]
    if opposed:
        return f"opposition from {', '.join(opposed[:2])}"
    themes: list[str] = []
    for c in clusters:
        themes.extend(c.get("key_themes", [])[:1])
    return ", ".join(themes[:3]) if themes else "segment-specific concerns"


async def _enrich_with_llm(
    state: SingularityState,
    template: list[DecisionOption],
) -> DecisionEngineResult | None:
    payload = {
        "query": state.query,
        "template_options": [
            {"type": o.type, "action": o.action, "expected_outcome": o.expected_outcome,
             "confidence": o.confidence}
            for o in template
        ],
        "deliberation": state.metrics.get("deliberation"),
        "prediction_market": state.metrics.get("prediction_market"),
        "monte_carlo": state.metrics.get("monte_carlo"),
        "swarm_optimization": state.metrics.get("swarm_optimization"),
    }
    user = DECISION_ENGINE_USER.format(data=json.dumps(payload, default=str))
    try:
        data = await get_ollama().generate_json(
            SINGULARITY_ENGINE_SYSTEM, user, temperature=0.4, max_tokens=900,
        )
        raw_opts = data.get("options", [])
        if not isinstance(raw_opts, list) or len(raw_opts) < 3:
            return None
        options: list[DecisionOption] = []
        seen_types: set[str] = set()
        for i, raw in enumerate(raw_opts):
            if not isinstance(raw, dict):
                continue
            opt_type = str(raw.get("type", _OPTION_TYPES[min(i, 4)]))
            if opt_type not in _OPTION_TYPES:
                opt_type = _OPTION_TYPES[min(i, 4)]
            if opt_type in seen_types:
                continue
            seen_types.add(opt_type)
            tmpl = next((t for t in template if t.type == opt_type), template[min(i, 4)])
            options.append(DecisionOption(
                type=opt_type,
                action=str(raw.get("action", tmpl.action))[:400],
                expected_outcome=str(raw.get("expected_outcome", tmpl.expected_outcome))[:200],
                supporting_evidence=_as_str_list(raw.get("supporting_evidence"), tmpl.supporting_evidence),
                causal_drivers=_as_str_list(raw.get("causal_drivers"), tmpl.causal_drivers),
                risks=_as_str_list(raw.get("risks"), tmpl.risks),
                confidence=float(np.clip(float(raw.get("confidence", tmpl.confidence)), 0.0, 1.0)),
            ))
        for opt_type in _OPTION_TYPES:
            if opt_type not in seen_types:
                tmpl = next(t for t in template if t.type == opt_type)
                options.append(tmpl)
        options.sort(key=lambda o: _OPTION_TYPES.index(o.type))
        return DecisionEngineResult(options=options[:5])
    except Exception as exc:  # noqa: BLE001
        logger.debug("LLM decision enrich failed: %s", exc)
        return None


def _as_str_list(val: Any, fallback: list[str]) -> list[str]:
    if isinstance(val, list):
        return [str(v)[:160] for v in val][:5]
    return fallback
