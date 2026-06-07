"""Programmatic NLG + batched LLM rendering for sampled agents."""
from __future__ import annotations

import asyncio
import logging
import random
from typing import Any

import numpy as np

from agents.cognitive.types import AgentCognitiveOutput, CognitiveStateVector
from config import get_settings
from llm.ollama_client import get_ollama
from prompts import COGNITIVE_RESPONSE_SYSTEM, COGNITIVE_RESPONSE_USER

logger = logging.getLogger("singularity.cognitive.renderer")

_PHRASE_POSITIVE = [
    "Honestly, I like where this is going",
    "I'm leaning yes on this",
    "This could work for me",
    "I'm fairly positive about it",
]
_PHRASE_NEUTRAL = [
    "I'm on the fence about this",
    "Could go either way for me",
    "I need to sit with it a bit longer",
    "Not fully convinced yet",
]
_PHRASE_NEGATIVE = [
    "I'm not sold on this",
    "This doesn't feel right for me",
    "I'd probably pass for now",
    "Skeptical until I see more proof",
]
_HESITATION = [
    "but I'd wait for reviews first",
    "though I'd want to see how people around me react",
    "but the price would need to make sense",
    "though I'm not rushing into anything",
]
_FACET_HOOKS: dict[str, dict[str, str]] = {
    "Intellect": {"high": "I want to see the reasoning hold up", "low": "I don't need all the details"},
    "Trust": {"high": "I tend to trust the source here", "low": "I'd verify before believing it"},
    "Anxiety": {"high": "the downside is what's on my mind", "low": "I'm not too worried about risk"},
    "Adventurousness": {"high": "I'm open to trying something new", "low": "I'd stick with what's familiar"},
    "Gregariousness": {"high": "I'd talk this over with people first", "low": "I'd decide on my own"},
}


def render_programmatic(output: AgentCognitiveOutput, topic: str, rng: random.Random) -> str:
    """Natural first-person comment from deliberation output."""
    state = output.state
    s = output.sentiment

    if s >= 0.2:
        opener = rng.choice(_PHRASE_POSITIVE)
    elif s <= -0.2:
        opener = rng.choice(_PHRASE_NEGATIVE)
    else:
        opener = rng.choice(_PHRASE_NEUTRAL)

    parts = [opener]
    if state.uncertainty > 0.45 and rng.random() < 0.65:
        parts.append(rng.choice(_HESITATION))

    # Facet hook from winning deliberation voice or top facet
    top_facet = max(state.facets, key=lambda k: abs(state.facets[k] - 50.0), default="")
    if top_facet in _FACET_HOOKS:
        band = "high" if state.facets[top_facet] >= 66 else "low" if state.facets[top_facet] <= 33 else "moderate"
        if band != "moderate":
            hook = _FACET_HOOKS[top_facet].get(band)
            if hook and rng.random() < 0.55:
                parts.append(hook)

    if topic and topic != "this" and rng.random() < 0.35:
        parts.append(f"when it comes to {topic}")

    body = ", ".join(parts)
    if len(body) > 190:
        body = body[:187].rstrip() + "..."
    return body


def _llm_user_payload(output: AgentCognitiveOutput, stimulus: str, context: str) -> str:
    trace_lines = output.state.deliberation.summary_lines()
    debate = "\n".join(f"- {line}" for line in trace_lines) if trace_lines else "- (internal conflict minimal)"
    em = output.state.emotions.as_dict()
    biases = ", ".join(output.state.active_bias_ids()[:4]) or "none prominent"
    return COGNITIVE_RESPONSE_USER.format(
        stimulus=stimulus,
        context=context,
        sentiment=output.sentiment,
        intent=output.behavioral_intent,
        confidence=output.state.confidence,
        uncertainty=output.state.uncertainty,
        emotions=f"trust={em['trust']}, fear={em['fear']}, curiosity={em['curiosity']}, excitement={em['excitement']}",
        active_biases=biases,
        internal_debate=debate,
        winning_voice=output.state.deliberation.winning_voice,
    )


async def _llm_render_one(
    output: AgentCognitiveOutput,
    stimulus: str,
    context: str,
    ocean_scores: dict[str, float],
) -> str:
    system = COGNITIVE_RESPONSE_SYSTEM.format(
        O=round(ocean_scores["O"]),
        C=round(ocean_scores["C"]),
        E=round(ocean_scores["E"]),
        A=round(ocean_scores["A"]),
        N=round(ocean_scores["N"]),
    )
    user = _llm_user_payload(output, stimulus, context)
    temp = round(0.55 + output.state.total_entropy * 0.4 + output.state.ocean.N / 300.0, 2)
    try:
        data = await get_ollama().generate_json(
            system,
            user,
            temperature=temp,
            max_tokens=get_settings().cognitive_deliberation_max_tokens,
        )
        comment = str(data.get("comment", "")).strip()
        if comment and len(comment) >= 12:
            return comment[:220]
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM cognitive render failed for %s: %s", output.state.agent_id, exc)
    rng = random.Random(output.state.entropy_seed + 99)
    return render_programmatic(output, "this", rng)


def select_llm_sample_indices(
    outputs: list[AgentCognitiveOutput],
    visual_clusters: np.ndarray,
    sample_size: int,
) -> set[int]:
    """Stratified sample: clusters + high entropy + influencers."""
    n = len(outputs)
    if sample_size >= n:
        return set(range(n))

    chosen: set[int] = set()
    clusters = int(visual_clusters.max()) + 1 if len(visual_clusters) else 3
    per_cluster = max(1, sample_size // clusters)
    for c in range(clusters):
        idxs = [i for i in range(n) if int(visual_clusters[i]) == c]
        idxs.sort(key=lambda i: outputs[i].state.total_entropy, reverse=True)
        for i in idxs[:per_cluster]:
            chosen.add(i)

    # High entropy fill
    by_entropy = sorted(range(n), key=lambda i: outputs[i].state.total_entropy, reverse=True)
    for i in by_entropy:
        if len(chosen) >= sample_size:
            break
        chosen.add(i)

    # Influencers
    by_social = sorted(range(n), key=lambda i: outputs[i].state.social_position, reverse=True)
    for i in by_social[: max(5, sample_size // 15)]:
        if len(chosen) >= sample_size:
            break
        chosen.add(i)

    return set(list(chosen)[:sample_size])


async def render_population_comments(
    outputs: list[AgentCognitiveOutput],
    llm_indices: set[int],
    stimulus: str,
    context: str,
    topic: str,
    run_seed: int = 0,
) -> None:
    """Fill comment + response_source on all outputs."""
    settings = get_settings()
    sem = asyncio.Semaphore(settings.cognitive_llm_concurrency)

    async def render_llm(i: int) -> None:
        async with sem:
            out = outputs[i]
            ocean = out.state.ocean.model_dump()
            comment = await _llm_render_one(out, stimulus, context, ocean)
            out.comment = comment
            out.response_source = "llm"

    # Programmatic first
    for i, out in enumerate(outputs):
        if i in llm_indices:
            continue
        rng = random.Random(out.state.entropy_seed + run_seed)
        out.comment = render_programmatic(out, topic, rng)
        out.response_source = "programmatic"

    if llm_indices:
        await asyncio.gather(*(render_llm(i) for i in llm_indices))
