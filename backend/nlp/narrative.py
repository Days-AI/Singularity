"""BERTopic narrative clustering over persona comment corpus."""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from config import get_settings

logger = logging.getLogger("singularity.nlp.narrative")

_model = None
_model_failed = False


def _fit_blocking(comments: list[str]) -> list[dict[str, Any]]:
    settings = get_settings()
    if len(comments) < settings.bertopic_min_comments:
        return []

    global _model, _model_failed
    if _model_failed:
        return []

    try:
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer

        if _model is None:
            st = SentenceTransformer(settings.keybert_model_id, device="cpu")
            _model = BERTopic(embedding_model=st, verbose=False, calculate_probabilities=False)

        topics, _ = _model.fit_transform(comments)
        topic_info = _model.get_topic_info()
        label_map: dict[int, str] = {}
        for _, row in topic_info.iterrows():
            tid = int(row["Topic"])
            if tid == -1:
                label_map[tid] = "Mixed narratives"
            else:
                words = _model.get_topic(tid)
                label_map[tid] = " · ".join(w for w, _ in (words or [])[:4]) or f"Topic {tid}"

        groups: dict[str, list[int]] = {}
        for i, t in enumerate(topics):
            label = label_map.get(int(t), "Mixed narratives")
            groups.setdefault(label, []).append(i)

        clusters = []
        for label, idxs in sorted(groups.items(), key=lambda x: -len(x[1])):
            clusters.append({"label": label, "size": len(idxs), "indices": idxs[:50]})
        return clusters[:10]
    except Exception as exc:  # noqa: BLE001
        logger.warning("BERTopic fit failed: %s", exc)
        _model_failed = True
        return []


async def discover_narrative_topics(comments: list[str]) -> list[dict[str, Any]]:
    """Return BERTopic-derived cluster summaries or [] when unavailable."""
    if not get_settings().nlp_pipeline_enabled:
        return []
    clean = [c.strip() for c in comments if c and len(c.strip()) > 8]
    if not clean:
        return []
    return await asyncio.to_thread(_fit_blocking, clean)


def merge_narrative_clusters(
    existing: list[dict[str, Any]],
    bertopic_clusters: list[dict[str, Any]],
    sentiments: list[float],
) -> list[dict[str, Any]]:
    """Prefer BERTopic labels when available; preserve sentiment aggregates."""
    if not bertopic_clusters:
        return existing

    merged: list[dict[str, Any]] = []
    for bc in bertopic_clusters:
        idxs = bc.get("indices") or []
        sents = [sentiments[i] for i in idxs if i < len(sentiments)]
        mean_sent = round(sum(sents) / len(sents), 3) if sents else 0.0
        merged.append({
            "label": bc["label"],
            "size": bc["size"],
            "sentiment": mean_sent,
            "source": "bertopic",
        })
    return merged[:8] if merged else existing
