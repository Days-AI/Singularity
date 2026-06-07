"""Days234/personality-engine wrapper (spec section 6).

Loads the gated HF model exactly as the vendor snippet specifies:

    hf_hub_download(repo_id="Days234/personality-engine",
                    filename="inference_Days234.py", token=HF_TOKEN)
    from inference_Days234 import PersonalityEngine
    engine = PersonalityEngine()
    engine.predict(text) -> {"ocean": {...}, "facets": {...}}

The engine is used for BOTH baseline OCEAN scoring and validation of
LLM-generated persona text (bias check, spec 6.2). Loading is lazy and
thread-offloaded; any failure (no token, gated 401, missing torch) degrades to
a deterministic analytic estimator so the flow always proceeds.

Heavy/gated imports (torch, huggingface_hub, the downloaded module) are
performed lazily inside the loader by design: it keeps app startup fast and is
what makes graceful degradation possible when the gated repo is unreachable.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import os
import sys
import threading
from pathlib import Path

from config import get_settings
from state import OceanScores

logger = logging.getLogger("singularity.personality")

# 6 facets per OCEAN dimension (spec 6.1).
FACETS: dict[str, list[str]] = {
    "O": ["Imagination", "Artistic", "Emotionality", "Adventurousness", "Intellect", "Liberalism"],
    "C": ["Self-Efficacy", "Orderliness", "Dutifulness", "Achievement", "Self-Discipline", "Cautiousness"],
    "E": ["Friendliness", "Gregariousness", "Assertiveness", "Activity", "Excitement", "Cheerfulness"],
    "A": ["Trust", "Morality", "Altruism", "Cooperation", "Modesty", "Sympathy"],
    "N": ["Anxiety", "Anger", "Depression", "Self-Consciousness", "Immoderation", "Vulnerability"],
}

_TRAIT_LEXICON: dict[str, list[str]] = {
    "O": ["new", "idea", "explore", "creative", "curious", "art", "imagine", "novel", "innovat"],
    "C": ["plan", "organize", "reliable", "careful", "discipline", "detail", "responsible", "efficient"],
    "E": ["people", "social", "party", "talk", "energy", "excited", "outgoing", "friend", "meet"],
    "A": ["help", "kind", "trust", "cooperate", "care", "warm", "support", "share", "love"],
    "N": ["worry", "anxious", "stress", "afraid", "nervous", "angry", "sad", "fear", "concern"],
}


class _EngineHandle:
    """Process-wide singleton guarding the (expensive) engine load."""

    _instance: "_EngineHandle | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self.settings = get_settings()
        self.engine = None
        self.available = False
        self._tried = False

    @classmethod
    def get(cls) -> "_EngineHandle":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _load(self) -> None:
        if self._tried:
            return
        self._tried = True
        if self.settings.disable_personality_engine:
            logger.info("Personality engine disabled by config; using analytic fallback.")
            return
        if not self.settings.hf_token:
            logger.warning("HF_TOKEN not set; Days234 engine unavailable, using fallback.")
            return
        try:
            from huggingface_hub import hf_hub_download

            cache_dir = Path(__file__).resolve().parent.parent / ".engine_cache"
            cache_dir.mkdir(exist_ok=True)
            module_path = hf_hub_download(
                repo_id=self.settings.personality_repo_id,
                filename="inference_Days234.py",
                local_dir=str(cache_dir),
                token=self.settings.hf_token,
            )
            module_dir = str(Path(module_path).resolve().parent)
            if module_dir not in sys.path:
                sys.path.insert(0, module_dir)
            os.environ.setdefault("HF_TOKEN", self.settings.hf_token)

            from inference_Days234 import PersonalityEngine  # type: ignore

            self.engine = PersonalityEngine()
            self.available = True
            logger.info("Days234 personality engine loaded.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Personality engine load failed (%s); using analytic fallback.", exc)
            self.engine = None
            self.available = False

    def predict(self, text: str) -> tuple[OceanScores, dict[str, float]]:
        self._load()
        if self.available and self.engine is not None:
            try:
                result = self.engine.predict(text)
                return _normalize(result.get("ocean", {}), result.get("facets", {}))
            except Exception as exc:  # noqa: BLE001
                logger.warning("engine.predict failed (%s); falling back.", exc)
        return _analytic_predict(text)


async def predict(text: str) -> tuple[OceanScores, dict[str, float]]:
    """Async wrapper - runs blocking torch inference in a worker thread."""
    return await asyncio.to_thread(_EngineHandle.get().predict, text)


def is_engine_available() -> bool:
    handle = _EngineHandle.get()
    handle._load()
    return handle.available


# --- normalization & fallback ------------------------------------------------
_OCEAN_ALIASES = {
    "o": "O", "openness": "O",
    "c": "C", "conscientiousness": "C",
    "e": "E", "extraversion": "E", "extroversion": "E",
    "a": "A", "agreeableness": "A",
    "n": "N", "neuroticism": "N",
}


def _normalize(ocean_raw: dict, facets_raw: dict) -> tuple[OceanScores, dict[str, float]]:
    mapped: dict[str, float] = {}
    for key, val in (ocean_raw or {}).items():
        canon = _OCEAN_ALIASES.get(str(key).strip().lower())
        if canon and isinstance(val, (int, float)):
            mapped[canon] = float(val)
    if mapped and max(abs(v) for v in mapped.values()) <= 1.0:
        mapped = {k: v * 100.0 for k, v in mapped.items()}
    ocean = OceanScores(
        O=_clamp(mapped.get("O", 50.0)),
        C=_clamp(mapped.get("C", 50.0)),
        E=_clamp(mapped.get("E", 50.0)),
        A=_clamp(mapped.get("A", 50.0)),
        N=_clamp(mapped.get("N", 50.0)),
    )
    facets: dict[str, float] = {}
    for key, val in (facets_raw or {}).items():
        if isinstance(val, (int, float)):
            v = float(val)
            facets[str(key)] = _clamp(v * 100.0 if abs(v) <= 1.0 else v)
    if not facets:
        facets = _derive_facets(ocean)
    return ocean, facets


def _analytic_predict(text: str) -> tuple[OceanScores, dict[str, float]]:
    """Deterministic lexicon estimator used when the engine is unavailable."""
    t = (text or "").lower()
    tokens = max(1, len(t.split()))
    scores: dict[str, float] = {}
    seed = int(hashlib.sha256(t.encode()).hexdigest(), 16)
    for trait, words in _TRAIT_LEXICON.items():
        hits = sum(t.count(w) for w in words)
        density = hits / math.sqrt(tokens)
        jitter = ((seed >> (ord(trait) % 16)) % 21) - 10  # -10..10, stable per text
        scores[trait] = _clamp(50.0 + density * 28.0 + jitter)
    ocean = OceanScores(O=scores["O"], C=scores["C"], E=scores["E"], A=scores["A"], N=scores["N"])
    return ocean, _derive_facets(ocean)


def _derive_facets(ocean: OceanScores) -> dict[str, float]:
    out: dict[str, float] = {}
    base = {"O": ocean.O, "C": ocean.C, "E": ocean.E, "A": ocean.A, "N": ocean.N}
    for dim, names in FACETS.items():
        center = base[dim]
        for i, name in enumerate(names):
            offset = math.sin(i * 1.3) * 8.0
            out[name] = _clamp(center + offset)
    return out


def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return float(max(lo, min(hi, v)))
