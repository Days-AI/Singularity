"""Programmatic NLG + batched LLM rendering for sampled agents."""
from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import TYPE_CHECKING

import numpy as np

from agents.cognitive.types import AgentCognitiveOutput
from config import get_settings
from llm.ollama_client import get_ollama
from nlp.cognitive_draft import build_cognitive_draft
from nlp.diversity import sampling_params
from nlp.emotion import detect_emotion
from nlp.naturalize import naturalize_drafts_batch
from nlp.polish import polish_persona_comment
from nlp.prompt_budget import clamp_text, compact_debate_lines, fit_prompt_pair
from nlp.sentiment import sentiment_aligned
from nlp.style_engine import compose_programmatic_fallback, select_voice
from prompts import COGNITIVE_RESPONSE_SYSTEM, COGNITIVE_RESPONSE_USER

if TYPE_CHECKING:
    from nlp.topic_context import TopicContext

logger = logging.getLogger("singularity.cognitive.renderer")


def _llm_user_payload(
    output: AgentCognitiveOutput,
    stimulus: str,
    context: str,
    topic_ctx: TopicContext | None,
    voice_register: str,
) -> str:
    trace_lines = output.state.deliberation.summary_lines()
    debate = compact_debate_lines(trace_lines)
    em = output.state.emotions.as_dict()
    return COGNITIVE_RESPONSE_USER.format(
        context=clamp_text(context, 140),
        stimulus=clamp_text(stimulus, 180),
        domain_label=topic_ctx.domain_label if topic_ctx else "general",
        topic_keyphrases=clamp_text(
            topic_ctx.keyphrase_str(3) if topic_ctx else "", 80
        ),
        voice_register=voice_register,
        sentiment=output.sentiment,
        intent=clamp_text(output.behavioral_intent, 48),
        confidence=output.state.confidence,
        emotions=f"trust={em['trust']:.1f}, fear={em['fear']:.1f}, curiosity={em['curiosity']:.1f}",
        internal_debate=debate,
        winning_voice=clamp_text(output.state.deliberation.winning_voice, 40),
    )


def _emotion_from_cognitive(output: AgentCognitiveOutput) -> str:
    em = output.state.emotions
    pairs = [
        ("fear", em.fear),
        ("curiosity", em.curiosity),
        ("excitement", em.excitement),
        ("trust", em.trust),
    ]
    label, _ = max(pairs, key=lambda item: item[1])
    return label


async def _llm_render_one(
    output: AgentCognitiveOutput,
    stimulus: str,
    context: str,
    ocean_scores: dict[str, float],
    topic_ctx: TopicContext | None,
    voice_register: str,
) -> str:
    settings = get_settings()
    system = COGNITIVE_RESPONSE_SYSTEM.format(
        O=round(ocean_scores["O"]),
        C=round(ocean_scores["C"]),
        E=round(ocean_scores["E"]),
        A=round(ocean_scores["A"]),
        N=round(ocean_scores["N"]),
    )
    user = _llm_user_payload(output, stimulus, context, topic_ctx, voice_register)
    ctx = settings.cognitive_ollama_num_ctx or settings.ollama_num_ctx
    max_tok = settings.cognitive_deliberation_max_tokens
    _, user, _ = fit_prompt_pair(
        system,
        user,
        num_ctx=ctx,
        num_predict=max_tok,
    )
    params = sampling_params(
        output.state.total_entropy,
        output.state.ocean.N,
        voice_register,
    )
    allow_retry = not settings.skip_cognitive_nlg_extras
    try:
        data = await get_ollama().generate_json(
            system,
            user,
            temperature=params.temperature,
            top_p=params.top_p,
            repeat_penalty=params.repeat_penalty,
            max_tokens=max_tok,
            num_ctx=ctx,
        )
        comment = str(data.get("comment", "")).strip()
        if comment and len(comment) >= 12:
            if allow_retry and not sentiment_aligned(
                comment, output.sentiment, settings.vader_sentiment_max_delta
            ):
                retry_user = user + f"\nTone: match sentiment {output.sentiment:+.2f}."
                _, retry_user, _ = fit_prompt_pair(
                    system, retry_user, num_ctx=ctx, num_predict=max_tok
                )
                data2 = await get_ollama().generate_json(
                    system,
                    retry_user,
                    temperature=params.temperature,
                    top_p=params.top_p,
                    repeat_penalty=params.repeat_penalty,
                    max_tokens=max_tok,
                    num_ctx=ctx,
                )
                comment = str(data2.get("comment", comment)).strip()
            if comment and (
                allow_retry
                or sentiment_aligned(comment, output.sentiment, settings.vader_sentiment_max_delta)
            ):
                return comment[:220]
            if comment and not allow_retry:
                return comment[:220]
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM cognitive render failed for %s: %s", output.state.agent_id, exc)
    rng = random.Random(output.state.entropy_seed + 99)
    return compose_programmatic_fallback(output, topic_ctx, stimulus, rng)


def select_llm_sample_indices(
    outputs: list[AgentCognitiveOutput],
    visual_clusters: np.ndarray,
    sample_size: int,
) -> set[int]:
    """Stratified sample: clusters + sentiment quartiles + voice + entropy."""
    n = len(outputs)
    if sample_size >= n:
        return set(range(n))

    chosen: set[int] = set()
    clusters = int(visual_clusters.max()) + 1 if len(visual_clusters) else 3
    per_cluster = max(1, sample_size // max(clusters, 1))

    for c in range(clusters):
        idxs = [i for i in range(n) if int(visual_clusters[i]) == c]
        idxs.sort(key=lambda i: outputs[i].state.total_entropy, reverse=True)
        for i in idxs[:per_cluster]:
            chosen.add(i)

    by_sent = sorted(range(n), key=lambda i: outputs[i].sentiment)
    q = max(1, sample_size // 8)
    for i in by_sent[:q] + by_sent[-q:] + by_sent[len(by_sent) // 2 - q : len(by_sent) // 2 + q]:
        chosen.add(i)

    by_entropy = sorted(range(n), key=lambda i: outputs[i].state.total_entropy, reverse=True)
    for i in by_entropy:
        if len(chosen) >= sample_size:
            break
        chosen.add(i)

    by_social = sorted(range(n), key=lambda i: outputs[i].state.social_position, reverse=True)
    for i in by_social[: max(5, sample_size // 20)]:
        if len(chosen) >= sample_size:
            break
        chosen.add(i)

    return set(list(chosen)[:sample_size])


async def _render_naturalized_batch(
    outputs: list[AgentCognitiveOutput],
    indices: list[int],
    stimulus: str,
    topic: str,
    run_seed: int,
    topic_context: TopicContext | None,
) -> None:
    """Packed Gemma naturalize for all programmatic agents (Ollama semaphore gates concurrency)."""
    drafts = []
    meta: list[tuple[int, str]] = []
    for i in indices:
        out = outputs[i]
        p_idx = int(out.state.agent_id.split("_")[-1]) if "_" in out.state.agent_id else i
        rng = random.Random(out.state.entropy_seed + run_seed)
        voice = select_voice(out.state.ocean, p_idx, out.state.total_entropy, rng)
        out.voice_register = voice.register
        drafts.append(
            build_cognitive_draft(
                out, topic_context, stimulus, topic=topic, voice_register=voice.register
            )
        )
        meta.append((i, voice.register))

    results, _ = await naturalize_drafts_batch(drafts)
    for (i, _), (comment, src) in zip(meta, results, strict=True):
        out = outputs[i]
        out.comment = comment
        out.response_source = src
        out.detected_emotion = _emotion_from_cognitive(out)


async def render_population_comments(
    outputs: list[AgentCognitiveOutput],
    llm_indices: set[int],
    stimulus: str,
    context: str,
    topic: str,
    run_seed: int = 0,
    topic_context: TopicContext | None = None,
) -> None:
    """Fill comment, response_source, voice_register, detected_emotion on all outputs."""
    settings = get_settings()
    skip_extras = settings.skip_cognitive_nlg_extras
    programmatic_indices = [i for i in range(len(outputs)) if i not in llm_indices]

    llm_sem = asyncio.Semaphore(max(1, settings.cognitive_llm_concurrency))

    async def render_llm(i: int) -> None:
        async with llm_sem:
            out = outputs[i]
            p_idx = int(out.state.agent_id.split("_")[-1]) if "_" in out.state.agent_id else i
            rng = random.Random(out.state.entropy_seed)
            voice = select_voice(out.state.ocean, p_idx, out.state.total_entropy, rng)
            out.voice_register = voice.register
            ocean = out.state.ocean.model_dump()
            comment = await _llm_render_one(
                out, stimulus, context, ocean, topic_context, voice.register
            )
            if skip_extras:
                out.detected_emotion = _emotion_from_cognitive(out)
                out.comment = comment
                out.response_source = "llm"
                return
            emo = await detect_emotion(comment)
            if emo:
                out.detected_emotion = emo[0]
            else:
                out.detected_emotion = _emotion_from_cognitive(out)
            ocean_summary = f"O{ocean['O']} C{ocean['C']} E{ocean['E']} A{ocean['A']} N{ocean['N']}"
            polished, src = await polish_persona_comment(
                comment,
                ocean_summary=ocean_summary,
                topic=topic_context.keyphrase_str() if topic_context else topic,
                voice_register=voice.register,
                sentiment=out.sentiment,
                intent=out.behavioral_intent,
                agent_id=out.state.agent_id,
                entropy=out.state.total_entropy,
                confidence=out.state.confidence,
            )
            out.comment = polished
            out.response_source = src

    t0 = time.monotonic()
    tasks: list[asyncio.Task[None]] = []
    if programmatic_indices:
        tasks.append(
            asyncio.create_task(
                _render_naturalized_batch(
                    outputs,
                    programmatic_indices,
                    stimulus,
                    topic,
                    run_seed,
                    topic_context,
                )
            )
        )
    if llm_indices:
        tasks.extend(asyncio.create_task(render_llm(i)) for i in llm_indices)
    await asyncio.gather(*tasks)

    fallback = sum(1 for o in outputs if o.response_source == "draft_fallback")
    naturalized = sum(1 for o in outputs if o.response_source == "naturalized")
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    logger.info(
        "Cognitive NLG complete: %d Gemma full / %d naturalized / %d fallback / %d agents (%dms, mode=%s)",
        len(llm_indices),
        naturalized,
        fallback,
        len(outputs),
        elapsed_ms,
        settings.latency_mode_normalized,
    )
    try:
        from observability.master_log import log_entry

        log_entry(
            "algo",
            "psychometric",
            "naturalize_complete",
            data={
                "count": len(programmatic_indices),
                "llm_sample_count": len(llm_indices),
                "naturalized": naturalized,
                "fallback_count": fallback,
                "elapsed_ms": elapsed_ms,
                "pack_size": settings.naturalize_pack_size,
            },
        )
    except Exception:  # noqa: BLE001
        pass
