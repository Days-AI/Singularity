"""Orchestrate full cognitive cycle for the population."""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

import numpy as np

from agents import personality
from agents.cognitive.bias_network import activate_biases
from agents.cognitive.deliberation import deliberate
from agents.cognitive.emotion_engine import init_emotions, update_emotions_after_response
from agents.cognitive.entropy import compute_entropy_seed, total_entropy
from agents.cognitive.facet_activator import facet_activation_weights
from agents.cognitive.memory_store import generate_memory, update_memory_after_stimulus
from agents.cognitive.response_renderer import render_population_comments, select_llm_sample_indices
from agents.cognitive.social_influence import (
    apply_social_influence,
    compute_social_position,
    identify_influencers,
    refresh_action_from_sentiment,
)
from agents.cognitive.types import AgentCognitiveOutput, CognitiveStateVector
from config import get_settings
from state import EvidenceItem, FacetScore, OceanScores, PersonaResponse

_FACET_ORDER: list[str] = [f for facets in personality.FACETS.values() for f in facets]


def _band(score: float) -> str:
    return "high" if score >= 66 else "low" if score <= 33 else "moderate"


def _salient_facets(facet_vec: np.ndarray) -> list[FacetScore]:
    order = sorted(
        range(len(_FACET_ORDER)),
        key=lambda i: abs(float(facet_vec[i]) - 50.0),
        reverse=True,
    )
    out: list[FacetScore] = []
    for i in order[:3]:
        score = round(float(facet_vec[i]), 1)
        out.append(FacetScore(name=_FACET_ORDER[i], score=score, band=_band(score)))
    return out


def _evidence_polarity(evidence: list[EvidenceItem]) -> float:
    if not evidence:
        return 0.0
    vals = [e.sentiment for e in evidence if e.sentiment is not None]
    if not vals:
        return 0.0
    return float(np.mean(vals))


@dataclass
class PopulationCognitiveResult:
    outputs: list[AgentCognitiveOutput] = field(default_factory=list)
    social_contagion_index: float = 0.0
    entropy_mean: float = 0.0
    llm_sample_count: int = 0


async def run_population_cognitive(
    *,
    population: int,
    pop_ocean: np.ndarray,
    pop_facets: np.ndarray,
    pop_sent: np.ndarray,
    archetype_assign: np.ndarray,
    responses: list[PersonaResponse],
    visual_clusters: np.ndarray,
    cluster_labels: dict[int, str],
    query: str,
    evidence: list[EvidenceItem],
    context: str,
    stimulus: str,
    topic: str,
    run_seed: int | None = None,
) -> PopulationCognitiveResult:
    settings = get_settings()
    if run_seed is None:
        run_seed = int(time.time() * 1000) % 100000

    evidence_pol = _evidence_polarity(evidence)
    outputs: list[AgentCognitiveOutput] = []

    for p_idx in range(population):
        arch_idx = int(archetype_assign[p_idx]) % len(responses)
        arch = responses[arch_idx]
        o = pop_ocean[p_idx]
        ocean = OceanScores(
            O=round(float(o[0]), 1),
            C=round(float(o[1]), 1),
            E=round(float(o[2]), 1),
            A=round(float(o[3]), 1),
            N=round(float(o[4]), 1),
        )
        facets = {name: round(float(pop_facets[p_idx, fi]), 1) for fi, name in enumerate(_FACET_ORDER)}
        memory = generate_memory(p_idx, arch.archetype_id)
        social_pos = compute_social_position(pop_ocean[p_idx], pop_facets[p_idx], _FACET_ORDER)
        vcluster = int(visual_clusters[p_idx])
        label = cluster_labels.get(vcluster, "Pragmatists")

        state = CognitiveStateVector(
            agent_id=f"p_{p_idx:04d}",
            ocean=ocean,
            facets=facets,
            memory=memory,
            social_position=social_pos,
            entropy_seed=compute_entropy_seed(p_idx, run_seed),
            cluster=vcluster,
            cluster_label=label,
            archetype_id=arch.archetype_id,
        )
        state.emotions = init_emotions(ocean, facets, memory)
        state.active_biases = activate_biases(state)
        state.total_entropy = total_entropy(state, evidence_pol)

        facet_weights = facet_activation_weights(query, evidence, facets)
        rng = random.Random(state.entropy_seed)

        sentiment, confidence, uncertainty, concerns, intent, trace = deliberate(
            state, facet_weights, evidence_pol, rng
        )
        state.confidence = confidence
        state.uncertainty = uncertainty
        state.deliberation = trace

        action = max(0.0, min(1.0, 0.5 + sentiment * 0.35 + (50.0 - ocean.N) / 400.0))

        outputs.append(
            AgentCognitiveOutput(
                state=state,
                sentiment=sentiment,
                behavioral_intent=intent,
                key_concerns=concerns,
                action_likelihood=round(action, 3),
            )
        )

    # Social influence pass
    influencers = identify_influencers(outputs, pop_ocean, pop_facets, _FACET_ORDER)
    contagion = apply_social_influence(outputs, visual_clusters, influencers)
    for out in outputs:
        if out.state.social_influence_received > 0:
            refresh_action_from_sentiment(out)

    # Post-response state updates
    for out in outputs:
        update_memory_after_stimulus(out.state.memory, out.sentiment, query)
        out.state.emotions = update_emotions_after_response(
            out.state.emotions, out.sentiment, out.state.active_biases, out.state.memory
        )

    # NLG: programmatic + LLM sample
    sample_size = min(settings.cognitive_llm_sample_size, population)
    llm_indices = select_llm_sample_indices(outputs, visual_clusters, sample_size)
    await render_population_comments(outputs, llm_indices, stimulus, context, topic, run_seed)

    # Sync pop_sent from cognitive sentiment
    for i, out in enumerate(outputs):
        pop_sent[i] = out.sentiment

    entropy_mean = float(np.mean([o.state.total_entropy for o in outputs])) if outputs else 0.0

    return PopulationCognitiveResult(
        outputs=outputs,
        social_contagion_index=contagion,
        entropy_mean=round(entropy_mean, 4),
        llm_sample_count=len(llm_indices),
    )


def cognitive_output_to_salient(output: AgentCognitiveOutput, pop_facets_row: np.ndarray) -> list[FacetScore]:
    return _salient_facets(pop_facets_row)
