"""IPIP-300 dataset loader (spec section 6.1).

Loads a real 1,500-row IPIP-300 score file (OCEAN dimensions + 30 facets, 0..100)
into numpy arrays aligned to the engine's canonical facet order. Returns None if
the file is absent or malformed so the psychometric engine can fall back to a
synthetic lattice.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from agents.personality import FACETS

logger = logging.getLogger("singularity.ipip")

_OCEAN_COLS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]

# CSV facet column names in the engine's canonical facet order (FACETS dict).
_FACET_COLS = [
    "facet_imagination", "facet_artistic_interests", "facet_emotionality",
    "facet_adventurousness", "facet_intellect", "facet_liberalism",
    "facet_self_efficacy", "facet_orderliness", "facet_dutifulness",
    "facet_achievement_striving", "facet_self_discipline", "facet_cautiousness",
    "facet_friendliness", "facet_gregariousness", "facet_assertiveness",
    "facet_activity_level", "facet_excitement_seeking", "facet_cheerfulness",
    "facet_trust", "facet_morality", "facet_altruism", "facet_cooperation",
    "facet_modesty", "facet_sympathy",
    "facet_anxiety", "facet_anger", "facet_depression",
    "facet_self_consciousness", "facet_immoderation", "facet_vulnerability",
]
_FACET_NAMES = [f for facets in FACETS.values() for f in facets]


@dataclass
class IpipPopulation:
    ocean: np.ndarray   # (N, 5) columns O,C,E,A,N (0..100)
    facets: np.ndarray  # (N, 30) aligned to FACETS order (0..100)

    @property
    def size(self) -> int:
        return self.ocean.shape[0]


def load(path: str | None) -> IpipPopulation | None:
    if not path:
        return None
    p = Path(path)
    if p.is_dir():  # accept a directory: pick the first matching CSV
        candidates = sorted(p.glob("*IPIP*SCORES*.csv")) or sorted(p.glob("*.csv"))
        if not candidates:
            logger.warning("No IPIP CSV found under %s", p)
            return None
        p = candidates[0]
    if not p.is_file():
        logger.warning("IPIP data path not found: %s", p)
        return None
    try:
        import pandas as pd

        df = pd.read_csv(p)
        missing = [c for c in _OCEAN_COLS + _FACET_COLS if c not in df.columns]
        if missing:
            logger.warning("IPIP CSV missing columns %s; ignoring file.", missing[:5])
            return None
        ocean = df[_OCEAN_COLS].to_numpy(dtype=float)
        facets = df[_FACET_COLS].to_numpy(dtype=float)
        ocean = np.clip(np.nan_to_num(ocean, nan=50.0), 0, 100)
        facets = np.clip(np.nan_to_num(facets, nan=50.0), 0, 100)
        logger.info("Loaded %d IPIP-300 profiles from %s", len(df), p.name)
        return IpipPopulation(ocean=ocean, facets=facets)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load IPIP CSV %s: %s", p, exc)
        return None
