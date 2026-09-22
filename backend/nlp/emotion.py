"""DistilRoBERTa emotion detection for LLM persona comments."""
from __future__ import annotations

import asyncio
import logging
import threading

from config import get_settings

logger = logging.getLogger("singularity.nlp.emotion")

_pipeline = None
_load_lock = threading.Lock()
_load_failed = False


def _load_pipeline():
    global _pipeline, _load_failed
    if _load_failed:
        return None
    if _pipeline is not None:
        return _pipeline
    with _load_lock:
        if _load_failed or _pipeline is not None:
            return _pipeline
        try:
            from transformers import pipeline as hf_pipeline

            model_id = get_settings().emotion_model_id
            logger.info("Loading emotion model %s", model_id)
            loaded = hf_pipeline(
                "text-classification",
                model=model_id,
                top_k=1,
                device=-1,
            )
            _pipeline = loaded
        except Exception as exc:  # noqa: BLE001
            logger.warning("Emotion model load failed: %s", exc)
            _load_failed = True
            _pipeline = None
    return _pipeline


def emotion_available() -> bool:
    return _load_pipeline() is not None


def _classify_blocking(text: str) -> tuple[str, float] | None:
    pipe = _load_pipeline()
    if not pipe or not text.strip():
        return None
    try:
        out = pipe(text[:512])[0]
        if isinstance(out, list):
            out = out[0]
        label = str(out.get("label", "")).lower()
        score = float(out.get("score", 0.0))
        return label, score
    except Exception as exc:  # noqa: BLE001
        logger.debug("emotion classify failed: %s", exc)
        return None


async def detect_emotion(text: str) -> tuple[str, float] | None:
    return await asyncio.to_thread(_classify_blocking, text)
