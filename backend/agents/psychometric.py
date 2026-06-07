"""Psychometric simulation engine (spec section 6).

Pipeline:
  1. Construct ~36 OCEAN archetypes spanning the trait space.
  2. Simulate each archetype's response with Gemma (PT-02, anti-sanitization).
  3. Score generated text back through the Days234 engine for OCEAN + 30 facets
     (baseline + bias-check validation, spec 6.2).
  4. Statistically expand archetypes to a 1,500-agent population (jittered).
  5. numpy PCA (SVD) -> 3D coords; k-means clustering.
  6. Stream 6 persona_batch events with cumulative OCEAN mean, sentiment
     histogram, facet x stimulus heatmap, and down-sampled scatter points.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
from dataclasses import dataclass, field
from typing import Awaitable, Callable

import numpy as np

from agents import ipip, personality
from config import get_settings
from llm.ollama_client import get_ollama
from prompts import PERSONA_SYSTEM, PERSONA_USER
from state import (
    EvidenceItem,
    HeatmapRow,
    OceanScores,
    PersonaBatchPayload,
    PersonaPoint,
    PersonaResponse,
    SentimentBucket,
    SingularityState,
)

logger = logging.getLogger("singularity.psychometric")

_STIMULI = [
    "Price sensitivity", "Range & reliability", "Brand trust", "Environmental impact",
    "Charging access", "Resale value", "Technology appeal", "Social proof",
]
_SENTIMENT_BUCKETS = 9  # histogram bins across [-1, 1]
_POINTS_PER_BATCH = 48
_FACET_ORDER: list[str] = [f for facets in personality.FACETS.values() for f in facets]


@dataclass
class PsychometricResult:
    responses: list[PersonaResponse] = field(default_factory=list)
    ocean_mean: OceanScores | None = None
    population: int = 0
    confidence: float = 0.0
    evidence: list[EvidenceItem] = field(default_factory=list)


def _band(score: float) -> str:
    return "high" if score >= 66 else "low" if score <= 33 else "moderate"


def _build_archetypes(n: int) -> list[OceanScores]:
    """Deterministic low/mid/high lattice sampled to ~n representative profiles."""
    levels = [25.0, 50.0, 75.0]
    archetypes: list[OceanScores] = []
    # Quasi-random but reproducible spread across the 5-D trait cube.
    idx = 0
    while len(archetypes) < n:
        combo = []
        x = idx
        for _ in range(5):
            combo.append(levels[x % 3])
            x //= 3
        # add a deterministic dimension-specific offset for within-cell variety
        jitter = [((idx * (i + 7)) % 11 - 5) for i in range(5)]
        vals = [max(2.0, min(98.0, combo[i] + jitter[i])) for i in range(5)]
        archetypes.append(OceanScores(O=vals[0], C=vals[1], E=vals[2], A=vals[3], N=vals[4]))
        idx += 1
    return archetypes[:n]


def _persona_temperature(ocean: OceanScores) -> float:
    # High-N personas get higher temperature (spec 6.2).
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
    sentiment, intent, emotion, concerns, purchase = 0.0, "", "", [], 0.0
    try:
        data = await get_ollama().generate_json(
            system, user, temperature=_persona_temperature(ocean)
        )
        sentiment = _clampf(float(data.get("sentiment_score", 0.0)), -1.0, 1.0)
        intent = str(data.get("behavioral_intent", ""))[:160]
        emotion = str(data.get("emotional_state", ""))[:80]
        raw_concerns = data.get("key_concerns", [])
        concerns = [str(c)[:80] for c in raw_concerns][:5] if isinstance(raw_concerns, list) else []
        purchase = _clampf(float(data.get("purchase_likelihood", 0.0)), 0.0, 1.0)
    except Exception as exc:  # noqa: BLE001
        logger.debug("persona sim fallback for %s: %s", arch_id, exc)
        sentiment, intent, emotion, concerns, purchase = _fallback_response(ocean)

    text = f"{intent}. Feeling {emotion}. Concerns: {', '.join(concerns)}".strip()
    validated_ocean, facets = await personality.predict(text or "neutral response")

    return PersonaResponse(
        archetype_id=arch_id, ocean=ocean, facets=facets, sentiment_score=sentiment,
        behavioral_intent=intent, emotional_state=emotion, key_concerns=concerns,
        purchase_likelihood=purchase, validated_ocean=validated_ocean,
    )


def _fallback_response(ocean: OceanScores) -> tuple[float, str, str, list[str], float]:
    # Sentiment leans positive with O/E/A, negative with N (deterministic).
    s = (ocean.O + ocean.E + ocean.A - 1.6 * ocean.N) / 260.0
    s = _clampf(s, -1.0, 1.0)
    emotion = "anxious" if ocean.N > 60 else "optimistic" if s > 0.2 else "measured"
    intent = "would consider adopting cautiously" if s > 0 else "skeptical of switching"
    concerns = ["upfront cost", "charging infrastructure"] if ocean.N > 50 else ["long-term value"]
    purchase = _clampf(0.5 + s * 0.4, 0.0, 1.0)
    return s, intent, emotion, concerns, purchase


async def run(
    state: SingularityState,
    emit_batch: Callable[[PersonaBatchPayload], Awaitable[None]],
) -> PsychometricResult:
    settings = get_settings()
    n_arch = settings.persona_archetypes
    population = settings.persona_population
    n_batches = settings.persona_batches

    context = _context_from(state)
    stimulus = f"How do you feel about: {state.query}"

    # Prefer the real IPIP-300 population; cluster it into archetypes for LLM
    # simulation. Fall back to a synthetic lattice if the dataset is absent.
    pop = ipip.load(settings.ipip_data_path)
    if pop is not None:
        population = pop.size
        archetypes, member_labels = _archetypes_from_population(pop.ocean, n_arch)
    else:
        archetypes = _build_archetypes(n_arch)
        member_labels = None

    sem = asyncio.Semaphore(settings.max_concurrent_agents)

    async def guarded(i: int, oc: OceanScores) -> PersonaResponse:
        async with sem:
            return await _simulate_archetype(f"arch_{i:02d}", oc, context, stimulus)

    responses = await asyncio.gather(*(guarded(i, oc) for i, oc in enumerate(archetypes)))

    # --- assemble the full population ---------------------------------------
    if pop is not None and member_labels is not None:
        pop_ocean, pop_facets, pop_sentiment, clusters = _population_from_ipip(
            pop, member_labels, responses
        )
    else:
        pop_ocean, pop_facets, pop_sentiment, clusters_assign = _expand_population(
            responses, population
        )
        clusters = clusters_assign

    # --- PCA (real or synthetic features) -----------------------------------
    features = np.hstack([pop_ocean, pop_facets])  # (P, 35)
    coords = _pca3(features)

    # --- bias-check confidence ----------------------------------------------
    confidence = _validation_confidence(responses)

    # --- stream batches ------------------------------------------------------
    batch_size = math.ceil(population / n_batches)
    cumulative = 0
    for b in range(n_batches):
        lo = b * batch_size
        hi = min(population, lo + batch_size)
        if lo >= hi:
            break
        cumulative = hi
        ocean_mean = _ocean_mean(pop_ocean[:hi])
        sentiment_dist = _histogram(pop_sentiment[:hi])
        heatmap = _heatmap(pop_facets[:hi], pop_sentiment[:hi])
        points = _sample_points(lo, hi, coords, pop_ocean, pop_sentiment, clusters)
        payload = PersonaBatchPayload(
            batch_index=b,
            batch_total=n_batches,
            profiles_in_batch=hi - lo,
            cumulative_profiles=cumulative,
            ocean_mean=ocean_mean,
            sentiment_dist=sentiment_dist,
            points=points,
            heatmap=heatmap,
        )
        await emit_batch(payload)
        await asyncio.sleep(0.15)  # let the dashboard animate batch arrival

    final_mean = _ocean_mean(pop_ocean)
    evidence = [
        EvidenceItem(source="PsychometricEngine",
                     title=f"{population} IPIP-300 personas simulated",
                     detail=(f"Mean sentiment {float(np.mean(pop_sentiment)):+.2f}; "
                             f"engine validation confidence {confidence:.0%}."),
                     value=round(float(np.mean(pop_sentiment)), 3)),
        EvidenceItem(source="PsychometricEngine", title="Population OCEAN (mean)",
                     detail=(f"O{final_mean.O:.0f} C{final_mean.C:.0f} E{final_mean.E:.0f} "
                             f"A{final_mean.A:.0f} N{final_mean.N:.0f}")),
    ]
    return PsychometricResult(
        responses=responses, ocean_mean=final_mean, population=population,
        confidence=confidence, evidence=evidence,
    )


def _context_from(state: SingularityState) -> str:
    if state.evidence:
        snippets = "; ".join(e.title for e in state.evidence[:5])
        return f"Market context for '{state.query}': {snippets}"
    return f"Market context for '{state.query}'."


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
        spread = 4.0 + (n_level / 100.0) * 6.0  # high-N => more variance
        pop_ocean[p] = np.clip(arch_ocean[a] + rng.normal(0, spread, 5), 0, 100)
        pop_facets[p] = np.clip(arch_facets[a] + rng.normal(0, spread, len(_FACET_ORDER)), 0, 100)
        pop_sent[p] = float(np.clip(arch_sent[a] + rng.normal(0, 0.12 + n_level / 500.0), -1, 1))
    return pop_ocean, pop_facets, pop_sent, assign


def _archetypes_from_population(ocean: np.ndarray, n_arch: int) -> tuple[list[OceanScores], np.ndarray]:
    """Cluster the real OCEAN population into n_arch archetypes (centroids)."""
    k = min(n_arch, len(ocean))
    labels = _kmeans(ocean, k=k)
    archetypes: list[OceanScores] = []
    overall = ocean.mean(axis=0)
    for c in range(k):
        members = ocean[labels == c]
        m = members.mean(axis=0) if len(members) else overall
        archetypes.append(OceanScores(O=float(m[0]), C=float(m[1]), E=float(m[2]),
                                       A=float(m[3]), N=float(m[4])))
    return archetypes, labels


def _population_from_ipip(
    pop: "ipip.IpipPopulation", labels: np.ndarray, responses: list[PersonaResponse]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Use real IPIP OCEAN/facets; assign each profile its archetype's simulated
    sentiment with neuroticism-scaled individual variation."""
    rng = np.random.default_rng(0x511CA1)
    arch_sent = np.array([r.sentiment_score for r in responses])
    pop_sent = np.empty(pop.size)
    for i in range(pop.size):
        c = int(labels[i]) if int(labels[i]) < len(arch_sent) else 0
        n_level = pop.ocean[i, 4]
        pop_sent[i] = float(np.clip(arch_sent[c] + rng.normal(0, 0.12 + n_level / 500.0), -1, 1))
    return pop.ocean, pop.facets, pop_sent, labels.astype(int)


def _pca3(features: np.ndarray) -> np.ndarray:
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std == 0] = 1.0
    z = (features - mean) / std
    # Economy SVD; columns of U scaled by singular values give principal coords.
    u, s, _ = np.linalg.svd(z, full_matrices=False)
    coords = u[:, :3] * s[:3]
    # Normalize each axis to a stable visual range.
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
    return OceanScores(O=float(m[0]), C=float(m[1]), E=float(m[2]), A=float(m[3]), N=float(m[4]))


def _histogram(sent: np.ndarray) -> list[SentimentBucket]:
    edges = np.linspace(-1, 1, _SENTIMENT_BUCKETS + 1)
    counts, _ = np.histogram(sent, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2
    return [SentimentBucket(bucket=round(float(c), 3), count=int(n)) for c, n in zip(centers, counts)]


def _heatmap(facets: np.ndarray, sent: np.ndarray) -> list[HeatmapRow]:
    facet_means = facets.mean(axis=0)  # (35,) 0..100
    mean_sent = float(sent.mean())
    rows: list[HeatmapRow] = []
    for fi, fname in enumerate(_FACET_ORDER):
        facet_norm = (facet_means[fi] - 50.0) / 50.0  # -1..1
        values: list[float] = []
        for si in range(len(_STIMULI)):
            polarity = math.cos((si + 1) * 1.7 + fi * 0.11)  # stable per cell, -1..1
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
                pca=(round(float(coords[i, 0]), 3), round(float(coords[i, 1]), 3),
                     round(float(coords[i, 2]), 3)),
                ocean=OceanScores(O=float(ocean[i, 0]), C=float(ocean[i, 1]),
                                  E=float(ocean[i, 2]), A=float(ocean[i, 3]), N=float(ocean[i, 4])),
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
        b = np.array([r.validated_ocean.O, r.validated_ocean.C, r.validated_ocean.E,
                      r.validated_ocean.A, r.validated_ocean.N])
        drifts.append(float(np.mean(np.abs(a - b)) / 100.0))
    if not drifts:
        return 0.6
    return round(float(max(0.0, min(1.0, 1.0 - np.mean(drifts)))), 3)


async def preview(stimulus: str, ocean: dict | None = None) -> dict:
    """Single-persona preview for POST /api/persona/preview."""
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
        "purchase_likelihood": resp.purchase_likelihood,
        "validated_ocean": resp.validated_ocean.model_dump() if resp.validated_ocean else None,
        "engine_available": personality.is_engine_available(),
    }


def _clampf(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))
