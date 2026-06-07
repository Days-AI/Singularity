"""Pytest configuration — ensure backend package is importable."""
from __future__ import annotations

import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
TESTS = Path(__file__).resolve().parent
for p in (BACKEND, TESTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

