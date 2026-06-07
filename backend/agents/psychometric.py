"""Psychometric simulation engine (spec section 6).

Pipeline:
  1. Construct ~36 OCEAN archetypes spanning the trait space.
  2. Simulate each archetype's response with Gemma (PT-02, anti-sanitization).
  3. Score generated text back through the Days234 engine for OCEAN + 30 facets.
  4. Statistically expand archetypes to a 1,500-agent population (jittered).
  5. Entropy-driven cognitive cycle: bias activation → deliberation → NLG.
  6. numpy PCA (SVD) -> 3D coords; k-means clustering.
  7. Stream 6 persona_batch events with opinions (250 each), aggregates, and scatter.
"""
from __future__ import annotations

import asyncio
import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import numpy as np

from agents import ipip, personality
from agents.cognitive.aggregate import aggregate_deliberation_metrics
from agents.cognitive.state_engine import (
    cognitive_output_to_salient,
    run_population_cognitive,
)
from config import get_settings
from observability.master_log import log_entry
from llm.ollama_client import get_ollama
from prompts import PERSONA_SYSTEM, PERSONA_USER
from state import (
    EvidenceItem,
    FacetScore,
    HeatmapRow,
    OceanScores,
    PersonaBatchPayload,
    PersonaOpinion,
    PersonaPoint,
    PersonaResponse,
    SingularityState,
)

logger = logging.getLogger("singularity.psychometric")

_GENERIC_STIMULI = [
    "Value perception",
    "Trust & credibility",
    "Change readiness",
    "Social proof",
    "Convenience & access",
    "Ethical alignment",
    "Novelty appeal",
    "Commitment level",
]
_CLUSTER_NAMES = ["Skeptics", "Pragmatists", "Enthusiasts"]
_POINTS_PER_BATCH = 48
_FACET_ORDER: list[str] = [f for facets in personality.FACETS.values() for f in facets]


@dataclass
class PsychometricResult:
    responses: list[PersonaResponse] = field(default_factory=list)
    opinions: list[PersonaOpinion] = field(default_factory=list)
    ocean_mean: OceanScores | None = None
    population: int = 0
    confidence: float = 0.0
    evidence: list[EvidenceItem] = field(default_factory=list)
    deliberation: dict[str, Any] = field(default_factory=dict)


def _band(score: float) -> str:
    return "high" if score >= 66 else "low" if score <= 33 else "moderate"


def _build_archetypes(n: int) -> list[OceanScores]:
    levels = [25.0, 50.0, 75.0]
    archetypes: list[OceanScores] = []
    idx = 0
    while len(archetypes) < n:
        combo = []
        x = idx
        for _ in range(5):
            combo.append(levels[x % 3])
            x //= 3
        jitter = [((idx * (i + 7)) % 11 - 5) for i in range(5)]
        vals = [max(2.0, min(98.0, combo[i] + jitter[i])) for i in range(5)]
        archetypes.append(OceanScores(O=vals[0], C=vals[1], E=vals[2], A=vals[3], N=vals[4]))
        idx += 1
    return archetypes[:n]


def _persona_temperature(ocean: OceanScores) -> float:
    return round(0.6 + (ocean.N / 100.0) * 0.35, 2)


async def _simulate_archetype(
    arch_id: str, ocean: OceanScores, context: str, stimulus: str
) -> PersonaResponse:
    system = PERSONA_SYSTEM.format(
        O_score=round(ocean.O), O_band=_band(ocean.O),
        C_score=round(ocean.C), C_band=_band(ocean.C),
        E_score=round(ocean.E), E_band=_band(ocean.E),
        A_score=round(ocean.A), A_band=_band(ocean.A),
        N_score=round(ocean.N), N_band=_band(ocean.N),
        context=context,
    )
    user = PERSONA_USER.format(stimulus=stimulus)
    sentiment, intent, emotion, concerns, action = 0.0, "", "", [], 0.0
    try:
        data = await get_ollama().generate_json(
            system, user, temperature=_persona_temperature(ocean),
            max_tokens=get_settings().persona_max_tokens,
        )
        sentiment = _clampf(float(data.get("sentiment_score", 0.0)), -1.0, 1.0)
        intent = str(data.get("behavioral_intent", ""))[:160]
        emotion = str(data.get("emotional_state", ""))[:80]
        raw_concerns = data.get("key_concerns", [])
        concerns = [str(c)[:80] for c in raw_concerns][:5] if isinstance(raw_concerns, list) else []
        raw_action = data.get("action_likelihood", data.get("purchase_likelihood", 0.0))
        action = _clampf(float(raw_action), 0.0, 1.0)
    except Exception as exc:  # noqa: BLE001
        logger.warning("persona sim fallback for %s: %s", arch_id, exc)
        sentiment, intent, emotion, concerns, action = _fallback_response(ocean)

    text = f"{intent}. Feeling {emotion}. Concerns: {', '.join(concerns)}".strip()
    validated_ocean, facets = await personality.predict(text or "neutral response")

    return PersonaResponse(
        archetype_id=arch_id, ocean=ocean, facets=facets, sentiment_score=sentiment,
        behavioral_intent=intent, emotional_state=emotion, key_concerns=concerns,
        action_likelihood=action, validated_ocean=validated_ocean,
    )


def _fallback_response(ocean: OceanScores) -> tuple[float, str, str, list[str], float]:
    s = (ocean.O + ocean.E + ocean.A - 1.6 * ocean.N) / 260.0
    s = _clampf(s, -1.0, 1.0)
    emotion = "anxious" if ocean.N > 60 else "optimistic" if s > 0.2 else "measured"
    intent = "would consider engaging cautiously" if s > 0 else "skeptical of change"
    concerns = ["uncertainty", "upfront cost"] if ocean.N > 50 else ["long-term value"]
    action = _clampf(0.5 + s * 0.4, 0.0, 1.0)
    return s, intent, emotion, concerns, action


def _programmatic_archetype_responses(
    pop: ipip.IpipPopulation,
    member_labels: np.ndarray,
    archetypes: list[OceanScores],
) -> list[PersonaResponse]:
    """Build cluster-seed responses without LLM when cognitive+IPIP handle quality."""
    responses: list[PersonaResponse] = []
    for i, ocean in enumerate(archetypes):
        sentiment, intent, emotion, concerns, action = _fallback_response(ocean)
        members = member_labels == i
        if members.any():
            facet_mean = pop.facets[members].mean(axis=0)
            facets = {name: float(facet_mean[j]) for j, name in enumerate(_FACET_ORDER)}
        else:
            facets = {}
        responses.append(
            PersonaResponse(
                archetype_id=f"arch_{i:02d}",
                ocean=ocean,
                facets=facets,
                sentiment_score=sentiment,
                behavioral_intent=intent,
                emotional_state=emotion,
                key_concerns=concerns,
                action_likelihood=action,
                validated_ocean=ocean,
            )
        )
    return responses


async def run(
    state: SingularityState,
    emit_batch: Callable[[PersonaBatchPayload], Awaitable[None]],
) -> PsychometricResult:
    settings = get_settings()
    n_arch = settings.persona_archetypes
    population = settings.persona_population
    n_batches = settings.persona_batches
    stimuli = _GENERIC_STIMULI

    context = _context_from(state)
    stimulus = f"How do you feel about: {state.query}"
    topic = _topic_keyword(state.query)

    pop = ipip.load(settings.ipip_data_path)
    if pop is not None:
        population = pop.size
        archetypes, member_labels = _archetypes_from_population(pop.ocean, n_arch)
        log_entry(
            "data",
            "ipip",
            "ipip_load",
            session_id=state.session_id,
            flow_uuid=state.flow_uuid,
            data={
                "source": "ipip",
                "population_size": population,
                "archetype_count": len(archetypes),
            },
        )
    else:
        archetypes = _build_archetypes(n_arch)
        member_labels = None

    skip_archetype_llm = (
        settings.cognitive_agents_enabled
        and pop is not None
        and member_labels is not None
    )

    if skip_archetype_llm:
        assert pop is not None and member_labels is not None
        responses = _programmatic_archetype_responses(pop, member_labels, archetypes)
        log_entry(
            "algo",
            "psychometric",
            "archetype_llm_skipped",
            session_id=state.session_id,
            flow_uuid=state.flow_uuid,
            data={
                "archetype_llm_skipped": True,
                "reason": "cognitive+ipip",
                "archetype_count": len(archetypes),
            },
        )
        logger.info(
            "Skipped %d archetype LLM calls (cognitive+IPIP fast path)",
            len(archetypes),
        )
    else:
        sem = asyncio.Semaphore(settings.max_concurrent_agents)

        async def guarded(i: int, oc: OceanScores) -> PersonaResponse:
            async with sem:
                return await _simulate_archetype(f"arch_{i:02d}", oc, context, stimulus)

        responses = await asyncio.gather(*(guarded(i, oc) for i, oc in enumerate(archetypes)))

    if pop is not None and member_labels is not None:
        pop_ocean, pop_facets, pop_sent, archetype_assign = _population_from_ipip(
            pop, member_labels, responses
        )
    else:
        pop_ocean, pop_facets, pop_sent, archetype_assign = _expand_population(
            responses, population
        )

    features = np.hstack([pop_ocean, pop_facets])
    coords = _pca3(features)
    visual_clusters = _kmeans(coords, k=3)
    cluster_labels = _cluster_label_map(visual_clusters, pop_sent)

    confidence = _validation_confidence(responses)
    all_opinions: list[PersonaOpinion] = []
    deliberation_metrics: dict[str, Any] = {}

    if settings.cognitive_agents_enabled:
        run_seed = settings.cognitive_run_seed
        cog = await run_population_cognitive(
            population=population,
            pop_ocean=pop_ocean,
            pop_facets=pop_facets,
            pop_sent=pop_sent,
            archetype_assign=archetype_assign,
            responses=responses,
            visual_clusters=visual_clusters,
            cluster_labels=cluster_labels,
            query=state.query,
            evidence=state.evidence,
            context=context,
            stimulus=stimulus,
            topic=topic,
            run_seed=run_seed,
        )
        all_opinions = _opinions_from_cognitive(cog.outputs, pop_ocean, pop_facets)
        deliberation_metrics = aggregate_deliberation_metrics(
            cog.outputs,
            responses,
            social_contagion_index=cog.social_contagion_index,
            entropy_mean=cog.entropy_mean,
        )
    else:
        all_opinions = _build_opinions_batch(
            0,
            population,
            responses,
            pop_ocean,
            pop_facets,
            pop_sent,
            archetype_assign,
            visual_clusters,
            cluster_labels,
            topic,
        )

    batch_size = math.ceil(population / n_batches)
    cumulative = 0
    for b in range(n_batches):
        lo = b * batch_size
        hi = min(population, lo + batch_size)
        if lo >= hi:
            break
        cumulative = hi
        ocean_mean = _ocean_mean(pop_ocean[:hi])
        heatmap = _heatmap(pop_facets[:hi], pop_sent[:hi], stimuli)
        points = _sample_points(lo, hi, coords, pop_ocean, pop_sent, visual_clusters)
        opinions = all_opinions[lo:hi]
        payload = PersonaBatchPayload(
            batch_index=b + 1,
            batch_total=n_batches,
            profiles_in_batch=hi - lo,
            cumulative_profiles=cumulative,
            ocean_mean=ocean_mean,
            points=points,
            heatmap=heatmap,
            opinions=opinions,
        )
        await emit_batch(payload)
        # Small yield to pace the streamed persona batches for the UI without
        # adding meaningful wall-clock latency to the run.
        await asyncio.sleep(0.05)

    final_mean = _ocean_mean(pop_ocean)
    mean_sent = float(np.mean(pop_sent))
    evidence = [
        EvidenceItem(
            source="PsychometricEngine",
            title=f"{population} IPIP-300 cognitive agents simulated",
            detail=(
                f"Mean sentiment {mean_sent:+.2f}; "
                f"engine validation confidence {confidence:.0%}."
                + (
                    f" Entropy mean {deliberation_metrics.get('entropy_mean', 0):.2f}; "
                    f"polarization {deliberation_metrics.get('polarization_index', 0):.2f}."
                    if deliberation_metrics
                    else ""
                )
            ),
            value=round(mean_sent, 3),
        ),
    ]
    return PsychometricResult(
        responses=responses,
        opinions=all_opinions,
        ocean_mean=final_mean,
        population=population,
        confidence=confidence,
        evidence=evidence,
        deliberation=deliberation_metrics,
    )


def _cluster_label_map(clusters: np.ndarray, sent: np.ndarray) -> dict[int, str]:
    k = int(clusters.max()) + 1 if len(clusters) else 3
    means = []
    for c in range(k):
        mask = clusters == c
        means.append((c, float(sent[mask].mean()) if mask.any() else 0.0))
    sorted_c = [c for c, _ in sorted(means, key=lambda x: x[1])]
    mapping: dict[int, str] = {}
    for i, c in enumerate(sorted_c[: len(_CLUSTER_NAMES)]):
        mapping[c] = _CLUSTER_NAMES[i]
    return mapping


def _opinions_from_cognitive(
    outputs: list,
    pop_ocean: np.ndarray,
    pop_facets: np.ndarray,
) -> list[PersonaOpinion]:
    opinions: list[PersonaOpinion] = []
    for i, out in enumerate(outputs):
        fields = out.to_opinion_fields()
        top_facets = cognitive_output_to_salient(out, pop_facets[i])
        src = fields.get("response_source")
        opinions.append(
            PersonaOpinion(
                id=out.state.agent_id,
                archetype_id=out.state.archetype_id,
                cluster=out.state.cluster,
                cluster_label=out.state.cluster_label,
                ocean=out.state.ocean,
                sentiment=fields["sentiment"],
                behavioral_intent=fields["behavioral_intent"],
                emotional_state=fields["emotional_state"],
                key_concerns=fields["key_concerns"],
                action_likelihood=fields["action_likelihood"],
                comment=fields["comment"],
                top_facets=top_facets,
                facets=fields.get("facets") or {},
                stance_confidence=fields.get("stance_confidence"),
                uncertainty=fields.get("uncertainty"),
                active_biases=fields.get("active_biases") or [],
                response_source=src if src in ("programmatic", "llm") else None,
            )
        )
    return opinions


def _build_opinions_batch(
    lo: int,
    hi: int,
    responses: list[PersonaResponse],
    pop_ocean: np.ndarray,
    pop_facets: np.ndarray,
    pop_sent: np.ndarray,
    archetype_assign: np.ndarray,
    visual_clusters: np.ndarray,
    cluster_labels: dict[int, str],
    topic: str,
) -> list[PersonaOpinion]:
    opinions: list[PersonaOpinion] = []
    for p_idx in range(lo, hi):
        arch_idx = int(archetype_assign[p_idx]) % len(responses)
        arch = responses[arch_idx]
        o = pop_ocean[p_idx]
        ocean = OceanScores(
            O=round(float(o[0]), 1), C=round(float(o[1]), 1),
            E=round(float(o[2]), 1), A=round(float(o[3]), 1), N=round(float(o[4]), 1),
        )
        sentiment = round(float(pop_sent[p_idx]), 3)
        vcluster = int(visual_clusters[p_idx])
        label = cluster_labels.get(vcluster, "Pragmatists")
        top_facets = _salient_facets(pop_facets[p_idx])
        opinions.append(
            PersonaOpinion(
                id=f"p_{p_idx:04d}",
                archetype_id=arch.archetype_id,
                cluster=vcluster,
                cluster_label=label,
                ocean=ocean,
                sentiment=sentiment,
                behavioral_intent=_vary_intent(arch.behavioral_intent, ocean, sentiment),
                emotional_state=_vary_emotion(arch.emotional_state, ocean),
                key_concerns=_vary_concerns(arch.key_concerns, ocean),
                action_likelihood=round(
                    _clampf(
                        arch.action_likelihood
                        + (sentiment - arch.sentiment_score) * 0.25
                        + (50.0 - ocean.N) / 400.0,
                        0.0,
                        1.0,
                    ),
                    3,
                ),
                comment=_persona_comment(
                    p_idx, ocean, top_facets, sentiment, label, topic,
                    arch.behavioral_intent, arch.emotional_state,
                ),
                top_facets=top_facets,
            )
        )
    return opinions


def _vary_intent(base: str, ocean: OceanScores, sentiment: float) -> str:
    if not base:
        base = "evaluating options carefully"
    if ocean.N > 65 and sentiment < 0:
        return f"{base}; hesitant given uncertainty"
    if ocean.E > 65 and sentiment > 0.2:
        return f"{base}; open to sharing views"
    if ocean.C > 65:
        return f"{base}; prefers structured next steps"
    return base[:160]


def _vary_emotion(base: str, ocean: OceanScores) -> str:
    if ocean.N > 60:
        return "cautiously apprehensive" if "anx" not in base.lower() else base[:80]
    if ocean.E > 60 and ocean.O > 55:
        return "curious and engaged" if not base else base[:80]
    return base[:80] if base else "measured"


def _vary_concerns(base: list[str], ocean: OceanScores) -> list[str]:
    concerns = list(base[:5]) if base else ["uncertainty"]
    if ocean.N > 55 and "risk" not in " ".join(concerns).lower():
        concerns.insert(0, "downside risk")
    if ocean.A > 60 and len(concerns) < 5:
        concerns.append("impact on others")
    return concerns[:5]


# --- first-person comment synthesis -----------------------------------------
# Per-persona first-person comments are generated programmatically so all
# 1,500 agents get a unique, OCEAN- and facet-grounded voice without 1,500 LLM
# calls. Generation is deterministic (seeded by persona index) for reproducible
# runs and zero repetition across reloads.

_STOPWORDS = {
    "analyze", "predict", "forecast", "market", "sentiment", "consumer",
    "quarter", "next", "about", "with", "from", "that", "this", "what", "when",
    "where", "which", "the", "and", "for", "behavioral", "drivers", "trends",
}

# Sentiment openers are a rare fallback only - used when a persona is uniformly
# moderate and no salient facet exists to anchor a reaction.
_SENTIMENT_OPENERS = {
    "positive": ["i'm cautiously into this", "leaning positive on it", "this could work for me"],
    "neutral": ["i'm on the fence", "could go either way", "need to sit with it"],
    "negative": ["not sold on this", "this feels off", "skeptical here"],
}

# Voice registers: stylistic wrappers chosen from the full OCEAN profile so two
# personas with similar salient facets still phrase reactions differently.
_REGISTER_PREFIX: dict[str, list[str]] = {
    "analyst": [
        "looking at it objectively,",
        "from what i can tell,",
        "weighing the facts,",
        "based on what i see,",
        "",
    ],
    "skeptic": [
        "i'm not fully convinced,",
        "i have reservations,",
        "something here gives me pause,",
        "my instinct is to push back,",
    ],
    "hype": [
        "this genuinely interests me,",
        "i find this compelling,",
        "this stands out to me,",
        "i'm drawn to the possibilities,",
    ],
    "pragmatist": [
        "practically speaking,",
        "bottom line for me,",
        "realistically,",
        "the practical view is,",
        "",
    ],
    "empath": ["personally,", "for me,", "i feel that", "from my standpoint,"],
    "blunt": ["to be direct,", "plainly,", "without sugarcoating,", "frankly,"],
    "casual": ["", "to be fair,", "from my perspective,", "in my view,"],
}

# Per-facet first-person reaction lines for all 30 IPIP facets, by band. Score
# magnitude buckets and the persona seed pick among the lines so high-85 and
# high-66 personas of the same facet can still phrase it differently.
_FACET_REACTIONS: dict[str, dict[str, list[str]]] = {
    # --- Openness ---
    "Imagination": {
        "high": ["my mind's already running with where this could go", "i keep picturing all the ways it plays out", "i can dream up a dozen uses for this"],
        "low": ["i'll stick to what's actually in front of me", "not one to daydream about it"],
        "moderate": ["i can see a couple angles here"],
    },
    "Artistic": {
        "high": ["the whole design of it really speaks to me", "aesthetically this just clicks for me"],
        "low": ["i don't care how polished it looks", "the look of it doesn't move me"],
        "moderate": ["the presentation's fine i guess"],
    },
    "Emotionality": {
        "high": ["this genuinely tugs at something for me", "i feel this one pretty deeply"],
        "low": ["i'm keeping my feelings out of it", "it doesn't hit me emotionally"],
        "moderate": ["a mild reaction overall"],
    },
    "Adventurousness": {
        "high": ["willing to try something different", "i love shaking up the routine for this", "new territory is exactly my thing"],
        "low": ["i'd rather stick with the safe option", "new territory makes me hesitate"],
        "moderate": ["open to it within reason"],
    },
    "Intellect": {
        "high": ["i want to see the actual data first", "show me the reasoning and i'm in", "i need the logic to hold up"],
        "low": ["don't need all the technical detail", "i'll skip the deep analysis"],
        "moderate": ["a quick rundown would do"],
    },
    "Liberalism": {
        "high": ["i'm all for challenging how it's usually done", "happy to break from how things have always been"],
        "low": ["i'd keep things the way they've always worked", "no need to upend the norm"],
        "moderate": ["depends how far it pushes things"],
    },
    # --- Conscientiousness ---
    "Self-Efficacy": {
        "high": ["confident i can make it work", "i know i'll handle this fine"],
        "low": ["not sure i could pull it off", "i doubt i'd manage it well"],
        "moderate": ["i could probably figure it out"],
    },
    "Orderliness": {
        "high": ["need the details lined up before i commit", "i want this organized first", "give me a clean plan and i'm in"],
        "low": ["i'm fine with the rough edges", "it doesn't need to be tidy for me"],
        "moderate": ["a loose plan works"],
    },
    "Dutifulness": {
        "high": ["if i say i'm in, i follow through", "i take the commitment seriously"],
        "low": ["i won't feel bound to stick with it", "i can walk away anytime"],
        "moderate": ["i'll honor it if it makes sense"],
    },
    "Achievement": {
        "high": ["this could really help me get ahead", "i'm chasing the payoff here", "i want the win this offers"],
        "low": ["not chasing anything big with this", "i'm not in it to win anything"],
        "moderate": ["a modest gain would be nice"],
    },
    "Self-Discipline": {
        "high": ["i can stay the course on this", "i won't lose focus partway through"],
        "low": ["i'd probably lose steam on it", "i struggle to stick with these"],
        "moderate": ["i'll keep at it for a while"],
    },
    "Cautiousness": {
        "high": ["want to wait and see first", "i'll think this through before acting", "i'm not rushing this decision"],
        "low": ["happy to just dive in", "i'll decide on the fly"],
        "moderate": ["i'll look before i leap, but not for long"],
    },
    # --- Extraversion ---
    "Friendliness": {
        "high": ["i'd happily bring others along", "warming up to it fast"],
        "low": ["i'll keep this to myself", "not the welcoming type on this"],
        "moderate": ["friendly enough about it"],
    },
    "Gregariousness": {
        "high": ["i want to talk this over with everyone", "this is better with a crowd weighing in"],
        "low": ["i'd rather decide solo", "i don't need the group for this"],
        "moderate": ["a couple opinions would help"],
    },
    "Assertiveness": {
        "high": ["i'll say it straight: here's my take", "i'm taking the lead on this one", "i'll make my position clear"],
        "low": ["i'll go with whatever the group picks", "i'd rather not push my view"],
        "moderate": ["i'll share if asked"],
    },
    "Activity": {
        "high": ["i want to get moving on this now", "no time to waste, let's go"],
        "low": ["no rush from me", "i'll get to it eventually"],
        "moderate": ["i'll pace myself on it"],
    },
    "Excitement": {
        "high": ["this genuinely excites me", "this lights me up a bit", "i'm energized about it"],
        "low": ["it doesn't excite me much", "i'm fairly unmoved by it"],
        "moderate": ["mildly interested"],
    },
    "Cheerfulness": {
        "high": ["feeling good about where this goes", "pretty upbeat on it"],
        "low": ["not exactly thrilled", "hard to feel cheery about this"],
        "moderate": ["cautiously okay with it"],
    },
    # --- Agreeableness ---
    "Trust": {
        "high": ["i trust the people behind it", "i'm inclined to take them at their word"],
        "low": ["not sure i trust the claims", "i'd verify before believing any of it", "the claims smell off to me"],
        "moderate": ["i only partly trust it"],
    },
    "Morality": {
        "high": ["it has to sit right ethically for me", "i won't cut corners on this"],
        "low": ["ethics aren't my main lens here", "i'm not fussed about the principles"],
        "moderate": ["as long as it's broadly fair"],
    },
    "Altruism": {
        "high": ["i care how this affects others", "i'd want everyone to benefit"],
        "low": ["i'm focused on what i get out of it", "others' outcomes aren't my concern here"],
        "moderate": ["i'd weigh the group somewhat"],
    },
    "Cooperation": {
        "high": ["happy to find common ground on it", "i'll work with whatever's agreed"],
        "low": ["i'll push back if i disagree", "i won't just go along to get along"],
        "moderate": ["i'll compromise where it counts"],
    },
    "Modesty": {
        "high": ["i'm not going to overstate my take", "i could easily be wrong here"],
        "low": ["i'm pretty sure my read is the right one", "my call is the one to trust"],
        "moderate": ["i think i'm mostly right"],
    },
    "Sympathy": {
        "high": ["i feel for whoever this impacts", "i can't ignore who might get hurt"],
        "low": ["the sob stories won't sway me", "feelings won't change my call"],
        "moderate": ["i note the human side"],
    },
    # --- Neuroticism ---
    "Anxiety": {
        "high": ["the what-ifs are eating at me", "i'm stressing about the downside", "honestly this makes me nervous"],
        "low": ["relatively calm about the risk", "not anxious about it at all"],
        "moderate": ["a little uneasy but ok"],
    },
    "Anger": {
        "high": ["this kind of irritates me", "i'm a bit fired up about it"],
        "low": ["nothing about it bothers me", "i'm not worked up either way"],
        "moderate": ["mildly annoyed at most"],
    },
    "Depression": {
        "high": ["somewhat pessimistic on this", "hard to feel hopeful here"],
        "low": ["staying upbeat about it", "i don't see a downside spiral"],
        "moderate": ["cautiously neutral on the outlook"],
    },
    "Self-Consciousness": {
        "high": ["i worry how i'd look choosing this", "what others think weighs on me"],
        "low": ["i don't care how it looks to anyone", "no self-image hang-ups here"],
        "moderate": ["a little image-conscious"],
    },
    "Immoderation": {
        "high": ["honestly might just impulse-jump on it", "i can't resist diving in"],
        "low": ["i can hold off no problem", "easy for me to wait"],
        "moderate": ["tempted but controlled"],
    },
    "Vulnerability": {
        "high": ["under pressure i might fold on this", "i could get overwhelmed by it"],
        "low": ["i'd stay steady under pressure", "stress won't shake my call"],
        "moderate": ["i'd manage the pressure okay"],
    },
}


def _sentiment_band(sentiment: float) -> str:
    if sentiment >= 0.15:
        return "positive"
    if sentiment <= -0.15:
        return "negative"
    return "neutral"


def _topic_keyword(query: str) -> str:
    words = [w.strip(".,?!:;\"'") for w in (query or "").split()]
    words = [w for w in words if len(w) > 3 and w.lower() not in _STOPWORDS]
    return " ".join(words[:3]) if words else "this"


def _salient_facets(facet_vec: np.ndarray) -> list[FacetScore]:
    """Top facets by absolute deviation from the neutral midpoint (50)."""
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


def _dominant_trait(ocean: OceanScores) -> tuple[str, str]:
    dims = {"O": ocean.O, "C": ocean.C, "E": ocean.E, "A": ocean.A, "N": ocean.N}
    dim = max(dims, key=lambda d: abs(dims[d] - 50.0))
    return dim, ("high" if dims[dim] >= 50.0 else "low")


def _voice_register(ocean: OceanScores, p_idx: int) -> str:
    """Pick a stylistic register from the full OCEAN profile, jittered by index
    so adjacent personas with similar profiles still differ."""
    candidates: list[str] = []
    if ocean.O >= 60 or ocean.C >= 65:
        candidates.append("analyst")
    if ocean.N >= 60 or ocean.A <= 40:
        candidates.append("skeptic")
    if ocean.E >= 62 and ocean.N <= 52:
        candidates.append("hype")
    if ocean.C >= 58:
        candidates.append("pragmatist")
    if ocean.A >= 62:
        candidates.append("empath")
    if ocean.E >= 58 and ocean.A <= 45:
        candidates.append("blunt")
    candidates.append("casual")
    return random.Random(p_idx * 7 + 13).choice(candidates)


def _facet_reaction(facet: FacetScore, p_idx: int, slot: int) -> str:
    """First-person clause driven by a facet's name + band + score magnitude."""
    bank = _FACET_REACTIONS.get(facet.name)
    if not bank:
        if facet.band == "high":
            return f"my {facet.name.lower()} really drives my take here"
        if facet.band == "low":
            return f"low {facet.name.lower()} means that's not what sways me"
        return ""
    lines = bank.get(facet.band) or []
    if not lines:
        return ""
    # Bucket by deviation magnitude so e.g. Anxiety 85 and 66 can diverge.
    bucket = int(abs(facet.score - 50.0) // 12)
    seed = p_idx * 31 + slot * 7 + (hash(facet.name) & 0xFFFF) + bucket
    return random.Random(seed).choice(lines)


def _persona_comment(
    p_idx: int,
    ocean: OceanScores,
    top_facets: list[FacetScore],
    sentiment: float,
    cluster_label: str,
    topic: str,
    arch_intent: str = "",
    arch_emotion: str = "",
) -> str:
    rng = random.Random(p_idx * 101 + 5)
    register = _voice_register(ocean, p_idx)

    # Facets are the PRIMARY voice driver. Prefer salient (non-moderate) facets,
    # falling back to whatever top facets exist.
    salient = [f for f in top_facets if f.band != "moderate"]
    primary_pool = salient or top_facets

    clauses: list[str] = []
    if primary_pool:
        clauses.append(_facet_reaction(primary_pool[0], p_idx, 0))
    if len(primary_pool) > 1 and rng.random() < 0.7:
        clauses.append(_facet_reaction(primary_pool[1], p_idx, 1))
    if len(primary_pool) > 2 and rng.random() < 0.3:
        clauses.append(_facet_reaction(primary_pool[2], p_idx, 2))
    clauses = [c for c in clauses if c]

    # Fallback only when a uniformly-moderate persona yielded no facet reaction.
    if not clauses:
        clauses = [rng.choice(_SENTIMENT_OPENERS[_sentiment_band(sentiment)])]

    # Weave in Gemma's archetype intent ~50% as one extra clause for texture.
    intent = (arch_intent or "").strip().rstrip(".")
    if intent and rng.random() < 0.5:
        lower = intent.lower()
        if not lower.startswith(("i ", "i'm", "im ", "my ")):
            intent = f"i {intent[0].lower()}{intent[1:]}"
        clauses.append(intent)

    # Topic anchor only sometimes, so it doesn't become a repeated tail.
    if topic and topic != "this" and rng.random() < 0.4:
        clauses.append(f"on {topic}")

    body = ", ".join(clauses)

    prefix = rng.choice(_REGISTER_PREFIX.get(register, [""]))
    if prefix:
        body = f"{prefix} {body}"

    if arch_emotion and rng.random() < 0.3:
        body = f"{body} ({arch_emotion})"

    body = body.strip().strip(",").strip()
    if len(body) > 190:
        body = body[:187].rstrip() + "..."
    return body


def _context_from(state: SingularityState) -> str:
    if state.evidence:
        snippets = "; ".join(e.title for e in state.evidence[:5])
        return f"Context for '{state.query}': {snippets}"
    return f"Context for '{state.query}'."


def _expand_population(
    responses: list[PersonaResponse], population: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0x511CA1)
    n_arch = len(responses)
    arch_ocean = np.array([[r.ocean.O, r.ocean.C, r.ocean.E, r.ocean.A, r.ocean.N] for r in responses])
    arch_facets = np.array([[r.facets.get(f, 50.0) for f in _FACET_ORDER] for r in responses])
    arch_sent = np.array([r.sentiment_score for r in responses])

    assign = np.array([i % n_arch for i in range(population)])
    rng.shuffle(assign)

    pop_ocean = np.empty((population, 5))
    pop_facets = np.empty((population, len(_FACET_ORDER)))
    pop_sent = np.empty(population)
    for p in range(population):
        a = assign[p]
        n_level = arch_ocean[a, 4]
        spread = 4.0 + (n_level / 100.0) * 6.0
        pop_ocean[p] = np.clip(arch_ocean[a] + rng.normal(0, spread, 5), 0, 100)
        pop_facets[p] = np.clip(arch_facets[a] + rng.normal(0, spread, len(_FACET_ORDER)), 0, 100)
        pop_sent[p] = float(np.clip(arch_sent[a] + rng.normal(0, 0.12 + n_level / 500.0), -1, 1))
    return pop_ocean, pop_facets, pop_sent, assign


def _archetypes_from_population(ocean: np.ndarray, n_arch: int) -> tuple[list[OceanScores], np.ndarray]:
    k = min(n_arch, len(ocean))
    labels = _kmeans(ocean, k=k)
    archetypes: list[OceanScores] = []
    overall = ocean.mean(axis=0)
    for c in range(k):
        members = ocean[labels == c]
        m = members.mean(axis=0) if len(members) else overall
        archetypes.append(
            OceanScores(
                O=float(m[0]), C=float(m[1]), E=float(m[2]),
                A=float(m[3]), N=float(m[4]),
            )
        )
    return archetypes, labels


def _population_from_ipip(
    pop: "ipip.IpipPopulation", labels: np.ndarray, responses: list[PersonaResponse]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0x511CA1)
    arch_sent = np.array([r.sentiment_score for r in responses])
    pop_sent = np.empty(pop.size)
    archetype_assign = labels.astype(int).copy()
    for i in range(pop.size):
        c = int(labels[i]) if int(labels[i]) < len(arch_sent) else 0
        n_level = pop.ocean[i, 4]
        pop_sent[i] = float(np.clip(arch_sent[c] + rng.normal(0, 0.12 + n_level / 500.0), -1, 1))
    return pop.ocean, pop.facets, pop_sent, archetype_assign


def _pca3(features: np.ndarray) -> np.ndarray:
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std == 0] = 1.0
    z = (features - mean) / std
    u, s, _ = np.linalg.svd(z, full_matrices=False)
    coords = u[:, :3] * s[:3]
    for j in range(coords.shape[1]):
        col = coords[:, j]
        rng = np.ptp(col) or 1.0
        coords[:, j] = (col - col.mean()) / rng * 6.0
    return coords


def _kmeans(coords: np.ndarray, k: int) -> np.ndarray:
    try:
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        return km.fit_predict(coords)
    except Exception as exc:  # noqa: BLE001
        logger.warning("sklearn KMeans unavailable (%s); using numpy fallback.", exc)
        return _kmeans_numpy(coords, k)


def _kmeans_numpy(coords: np.ndarray, k: int, iters: int = 25) -> np.ndarray:
    rng = np.random.default_rng(42)
    centroids = coords[rng.choice(len(coords), k, replace=False)]
    labels = np.zeros(len(coords), dtype=int)
    for _ in range(iters):
        d = np.linalg.norm(coords[:, None, :] - centroids[None, :, :], axis=2)
        labels = d.argmin(axis=1)
        for c in range(k):
            members = coords[labels == c]
            if len(members):
                centroids[c] = members.mean(axis=0)
    return labels


def _ocean_mean(ocean: np.ndarray) -> OceanScores:
    m = ocean.mean(axis=0)
    return OceanScores(
        O=float(m[0]), C=float(m[1]), E=float(m[2]), A=float(m[3]), N=float(m[4])
    )


def _heatmap(facets: np.ndarray, sent: np.ndarray, stimuli: list[str]) -> list[HeatmapRow]:
    facet_means = facets.mean(axis=0)
    mean_sent = float(sent.mean())
    rows: list[HeatmapRow] = []
    for fi, fname in enumerate(_FACET_ORDER):
        facet_norm = (facet_means[fi] - 50.0) / 50.0
        values: list[float] = []
        for si in range(len(stimuli)):
            polarity = math.cos((si + 1) * 1.7 + fi * 0.11)
            v = 0.5 * facet_norm * polarity + 0.4 * mean_sent + 0.1 * math.sin(fi + si)
            values.append(round(_clampf(v, -1.0, 1.0), 3))
        rows.append(HeatmapRow(facet=fname, values=values))
    return rows


def _sample_points(
    lo: int, hi: int, coords: np.ndarray, ocean: np.ndarray,
    sent: np.ndarray, clusters: np.ndarray,
) -> list[PersonaPoint]:
    idxs = np.linspace(lo, hi - 1, min(_POINTS_PER_BATCH, hi - lo)).astype(int)
    points: list[PersonaPoint] = []
    for i in idxs:
        points.append(
            PersonaPoint(
                id=f"p_{i:04d}",
                pca=(
                    round(float(coords[i, 0]), 3),
                    round(float(coords[i, 1]), 3),
                    round(float(coords[i, 2]), 3),
                ),
                ocean=OceanScores(
                    O=float(ocean[i, 0]), C=float(ocean[i, 1]),
                    E=float(ocean[i, 2]), A=float(ocean[i, 3]), N=float(ocean[i, 4]),
                ),
                sentiment=round(float(sent[i]), 3),
                cluster=int(clusters[i]),
            )
        )
    return points


def _validation_confidence(responses: list[PersonaResponse]) -> float:
    drifts = []
    for r in responses:
        if r.validated_ocean is None:
            continue
        a = np.array([r.ocean.O, r.ocean.C, r.ocean.E, r.ocean.A, r.ocean.N])
        b = np.array([
            r.validated_ocean.O, r.validated_ocean.C, r.validated_ocean.E,
            r.validated_ocean.A, r.validated_ocean.N,
        ])
        drifts.append(float(np.mean(np.abs(a - b)) / 100.0))
    if not drifts:
        return 0.6
    return round(float(max(0.0, min(1.0, 1.0 - np.mean(drifts)))), 3)


async def preview(stimulus: str, ocean: dict | None = None) -> dict:
    oc = OceanScores(
        O=float((ocean or {}).get("O", 60)), C=float((ocean or {}).get("C", 55)),
        E=float((ocean or {}).get("E", 50)), A=float((ocean or {}).get("A", 55)),
        N=float((ocean or {}).get("N", 45)),
    )
    resp = await _simulate_archetype("preview", oc, "Single persona preview.", stimulus)
    return {
        "ocean": resp.ocean.model_dump(),
        "facets": resp.facets,
        "sentiment_score": resp.sentiment_score,
        "behavioral_intent": resp.behavioral_intent,
        "emotional_state": resp.emotional_state,
        "key_concerns": resp.key_concerns,
        "action_likelihood": resp.action_likelihood,
        "validated_ocean": resp.validated_ocean.model_dump() if resp.validated_ocean else None,
        "engine_available": personality.is_engine_available(),
    }


def _clampf(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))
