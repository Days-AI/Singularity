"""Swarm intelligence optimization (spec stage 8).

Routes by query domain and applies ACO (communications), PSO (marketing),
or MACO (consumer research) to optimize narrative/creative/positioning paths.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from config import get_settings
from state import SingularityState

logger = logging.getLogger("singularity.swarm")

_COMM_KEYWORDS = re.compile(
    r"\b(pr|public.?relations|communications?|messaging|narrative|reputation|crisis|media)\b",
    re.I,
)
_MKT_KEYWORDS = re.compile(
    r"\b(marketing|campaign|advertis|creative|headline|brand|digital|social.?media|targeting)\b",
    re.I,
)
_PRODUCT_KEYWORDS = re.compile(
    r"\b(product|feature|concept|positioning|value.?prop|consumer|research|survey|adoption)\b",
    re.I,
)

_NARRATIVE_STEPS = [
    "awareness_hook",
    "credibility_proof",
    "emotional_resonance",
    "objection_handling",
    "call_to_action",
]
_CREATIVE_DIMS = ["headline_clarity", "visual_impact", "audience_fit", "urgency", "trust_signal"]
_POSITIONING_DIMS = ["differentiation", "price_value", "convenience", "social_proof", "innovation"]


@dataclass
class SwarmResult:
    domain: str = "consumer_research"
    algorithm: str = "maco"
    optimal_path: dict[str, Any] = field(default_factory=dict)
    alternatives: list[dict[str, Any]] = field(default_factory=list)
    convergence_iterations: int = 0


def run(state: SingularityState) -> SwarmResult:
    domain = _infer_domain(state)
    iterations = get_settings().swarm_iterations
    deliberation = state.metrics.get("deliberation", {})
    market = state.metrics.get("prediction_market", {})
    base_score = float(market.get("overall_outcome", 50.0)) / 100.0
    polarization = float(deliberation.get("polarization_index", 0.2))
    rng = np.random.default_rng(_seed(state.query))

    if domain == "communications":
        result = _run_aco(state, base_score, polarization, iterations, rng)
        algo = "aco"
    elif domain == "marketing":
        result = _run_pso(state, base_score, polarization, iterations, rng)
        algo = "pso"
    else:
        result = _run_maco(state, base_score, polarization, iterations, rng)
        algo = "maco"

    return SwarmResult(
        domain=domain,
        algorithm=algo,
        optimal_path=result["optimal"],
        alternatives=result["alternatives"],
        convergence_iterations=result["iterations"],
    )


def to_metrics(result: SwarmResult) -> dict[str, Any]:
    return {
        "domain": result.domain,
        "algorithm": result.algorithm,
        "optimal_path": result.optimal_path,
        "alternatives": result.alternatives,
        "convergence_iterations": result.convergence_iterations,
    }


def _infer_domain(state: SingularityState) -> str:
    text = state.query
    for e in state.evidence[:6]:
        text += " " + e.title + " " + e.detail
    if _COMM_KEYWORDS.search(text):
        return "communications"
    if _MKT_KEYWORDS.search(text):
        return "marketing"
    if _PRODUCT_KEYWORDS.search(text):
        return "consumer_research"
    return "consumer_research"


def _score_path(steps: list[str], base: float, polarization: float, rng: np.random.Generator) -> float:
    """Heuristic path scorer grounded in simulation signals."""
    n = len(steps)
    diversity_bonus = min(0.15, n * 0.02)
    polar_penalty = polarization * 0.1
    noise = rng.uniform(-0.03, 0.03)
    return float(np.clip(base + diversity_bonus - polar_penalty + noise, 0.0, 1.0))


def _run_aco(
    state: SingularityState,
    base: float,
    polarization: float,
    iterations: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    n_ants = 12
    pheromone = np.ones(len(_NARRATIVE_STEPS))
    best_steps: list[str] = list(_NARRATIVE_STEPS)
    best_score = 0.0

    for it in range(iterations):
        for _ in range(n_ants):
            order = list(_NARRATIVE_STEPS)
            rng.shuffle(order)
            score = _score_path(order, base, polarization, rng) * float(np.mean(pheromone))
            if score > best_score:
                best_score = score
                best_steps = order
            for j, step in enumerate(order):
                idx = _NARRATIVE_STEPS.index(step) if step in _NARRATIVE_STEPS else j
                pheromone[idx] = 0.9 * pheromone[idx] + 0.1 * score
        pheromone /= pheromone.sum()

    alts = []
    for _ in range(3):
        alt_order = list(_NARRATIVE_STEPS)
        rng.shuffle(alt_order)
        alts.append({
            "path": {"steps": alt_order},
            "score": round(_score_path(alt_order, base, polarization, rng), 4),
        })
    alts.sort(key=lambda x: -x["score"])

    return {
        "optimal": {"steps": best_steps, "score": round(best_score, 4)},
        "alternatives": alts[:3],
        "iterations": iterations,
    }


def _run_pso(
    state: SingularityState,
    base: float,
    polarization: float,
    iterations: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    n_particles = 15
    dim = len(_CREATIVE_DIMS)
    pos = rng.uniform(0.3, 0.9, (n_particles, dim))
    vel = rng.uniform(-0.05, 0.05, (n_particles, dim))
    pbest = pos.copy()
    pbest_scores = np.array([_particle_score(p, base, polarization) for p in pos])
    gbest_idx = int(np.argmax(pbest_scores))
    gbest = pbest[gbest_idx].copy()
    gbest_score = float(pbest_scores[gbest_idx])

    w, c1, c2 = 0.7, 1.5, 1.5
    for _ in range(iterations):
        for i in range(n_particles):
            r1, r2 = rng.random(dim), rng.random(dim)
            vel[i] = w * vel[i] + c1 * r1 * (pbest[i] - pos[i]) + c2 * r2 * (gbest - pos[i])
            pos[i] = np.clip(pos[i] + vel[i], 0.0, 1.0)
            score = _particle_score(pos[i], base, polarization)
            if score > pbest_scores[i]:
                pbest_scores[i] = score
                pbest[i] = pos[i].copy()
                if score > gbest_score:
                    gbest_score = score
                    gbest = pos[i].copy()

    steps = [f"{d}:{gbest[j]:.2f}" for j, d in enumerate(_CREATIVE_DIMS)]
    alts = []
    for i in np.argsort(-pbest_scores)[:3]:
        alt_steps = [f"{d}:{pbest[i][j]:.2f}" for j, d in enumerate(_CREATIVE_DIMS)]
        alts.append({"path": {"steps": alt_steps}, "score": round(float(pbest_scores[i]), 4)})

    return {
        "optimal": {"steps": steps, "score": round(gbest_score, 4)},
        "alternatives": alts,
        "iterations": iterations,
    }


def _particle_score(params: np.ndarray, base: float, polarization: float) -> float:
    mean_param = float(np.mean(params))
    balance = 1.0 - abs(mean_param - base) * 0.5
    polar_penalty = polarization * 0.08
    return float(np.clip(base * 0.6 + mean_param * 0.4 + balance * 0.1 - polar_penalty, 0.0, 1.0))


def _run_maco(
    state: SingularityState,
    base: float,
    polarization: float,
    iterations: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    deliberation = state.metrics.get("deliberation", {})
    archetypes = deliberation.get("persona_archetypes", [])
    n_agents = max(len(archetypes), 3)
    dim = len(_POSITIONING_DIMS)
    positions = rng.uniform(0.2, 0.8, (n_agents, dim))

    for _ in range(iterations):
        centroid = np.mean(positions, axis=0)
        for i in range(n_agents):
            noise = rng.normal(0, 0.02, dim)
            positions[i] = np.clip(positions[i] + 0.3 * (centroid - positions[i]) + noise, 0.0, 1.0)

    consensus = np.mean(positions, axis=0)
    score = _particle_score(consensus, base, polarization)
    steps = [f"{d}:{consensus[j]:.2f}" for j, d in enumerate(_POSITIONING_DIMS)]

    alts = []
    for i in range(min(3, n_agents)):
        alt_score = _particle_score(positions[i], base, polarization)
        alt_steps = [f"{d}:{positions[i][j]:.2f}" for j, d in enumerate(_POSITIONING_DIMS)]
        alts.append({"path": {"steps": alt_steps}, "score": round(alt_score, 4)})
    alts.sort(key=lambda x: -x["score"])

    return {
        "optimal": {"steps": steps, "score": round(score, 4)},
        "alternatives": alts[:3],
        "iterations": iterations,
    }


def _seed(query: str) -> int:
    return sum(ord(c) for c in query) % (2**31)
