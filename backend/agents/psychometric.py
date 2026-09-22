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
from nlp.cognitive_draft import build_draft_from_persona_fields
from nlp.naturalize import naturalize_drafts_batch
from nlp.style_engine import select_voice, topic_keyword
from nlp.prompt_budget import clamp_text, compact_nlg_context
from nlp.topic_context import TopicContext
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
    topic_context: TopicContext | None = None,
) -> PsychometricResult:
    settings = get_settings()
    n_arch = settings.persona_archetypes
    population = settings.persona_population
    n_batches = settings.persona_batches
    stimuli = _GENERIC_STIMULI

    context = _context_from(state, topic_context)
    hook = topic_context.evidence_hook() if topic_context else ""
    compact_stim, compact_ctx = compact_nlg_context(
        state.query,
        keyphrases=topic_context.keyphrases if topic_context else None,
        evidence_hook=hook,
    )
    stimulus = f"How do you feel about: {compact_stim}"
    if not context or context.startswith("Context for"):
        context = compact_ctx
    topic = topic_keyword(state.query)
    if topic_context and topic_context.keyphrases:
        topic = topic_context.keyphrases[0]

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
            topic_context=topic_context,
        )
        all_opinions = _opinions_from_cognitive(cog.outputs, pop_ocean, pop_facets)
        deliberation_metrics = aggregate_deliberation_metrics(
            cog.outputs,
            responses,
            social_contagion_index=cog.social_contagion_index,
            entropy_mean=cog.entropy_mean,
        )
        deliberation_metrics["llm_sample_count"] = cog.llm_sample_count
    else:
        all_opinions = await _build_opinions_batch_naturalized(
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
            stimulus,
            session_id=state.session_id,
            flow_uuid=state.flow_uuid,
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
                response_source=src if src in (
                    "programmatic", "llm", "llm_polished", "naturalized", "draft_fallback"
                ) else None,
                voice_register=fields.get("voice_register"),
                detected_emotion=fields.get("detected_emotion"),
            )
        )
    return opinions


async def _build_opinions_batch_naturalized(
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
    stimulus: str,
    *,
    session_id: str | None = None,
    flow_uuid: str | None = None,
) -> list[PersonaOpinion]:
    opinions: list[PersonaOpinion] = []
    drafts = []
    rows: list[dict] = []

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
        intent = _vary_intent(arch.behavioral_intent, ocean, sentiment)
        emotion = _vary_emotion(arch.emotional_state, ocean)
        concerns = _vary_concerns(arch.key_concerns, ocean)
        action = round(
            _clampf(
                arch.action_likelihood
                + (sentiment - arch.sentiment_score) * 0.25
                + (50.0 - ocean.N) / 400.0,
                0.0,
                1.0,
            ),
            3,
        )
        voice = select_voice(ocean, p_idx, entropy=0.5).register
        drafts.append(
            build_draft_from_persona_fields(
                agent_id=f"p_{p_idx:04d}",
                ocean=ocean,
                sentiment=sentiment,
                behavioral_intent=intent,
                emotional_state=emotion,
                key_concerns=concerns,
                action_likelihood=action,
                cluster_label=label,
                top_facets=top_facets,
                topic=topic,
                stimulus=stimulus,
                voice_register=voice,
            )
        )
        rows.append({
            "arch": arch,
            "ocean": ocean,
            "top_facets": top_facets,
            "sentiment": sentiment,
            "label": label,
            "vcluster": vcluster,
            "voice": voice,
            "intent": intent,
            "emotion": emotion,
            "concerns": concerns,
            "action": action,
        })

    results, _ = await naturalize_drafts_batch(
        drafts, session_id=session_id, flow_uuid=flow_uuid
    )

    for p_idx, row, (comment, src) in zip(range(lo, hi), rows, results, strict=True):
        arch = row["arch"]
        opinions.append(
            PersonaOpinion(
                id=f"p_{p_idx:04d}",
                archetype_id=arch.archetype_id,
                cluster=row["vcluster"],
                cluster_label=row["label"],
                ocean=row["ocean"],
                sentiment=row["sentiment"],
                behavioral_intent=row["intent"],
                emotional_state=row["emotion"],
                key_concerns=row["concerns"],
                action_likelihood=row["action"],
                comment=comment,
                top_facets=row["top_facets"],
                response_source=src if src in (
                    "naturalized", "llm_polished", "draft_fallback"
                ) else "naturalized",
                voice_register=row["voice"],
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


def _context_from(state: SingularityState, topic_context: TopicContext | None = None) -> str:
    if topic_context and topic_context.evidence_snippets:
        return clamp_text(topic_context.evidence_snippets[0], 140)
    if state.evidence:
        snippets = "; ".join(clamp_text(e.title, 80) for e in state.evidence[:3])
        return clamp_text(f"Context for '{state.query[:80]}': {snippets}", 200)
    return clamp_text(f"Context for '{state.query[:120]}'.", 140)


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
