"""Shared helpers for sync agent modules invoked from async flow."""
from __future__ import annotations

import asyncio
from typing import TypeVar

T = TypeVar("T")


def run_optional_async(coro) -> T | None:
    """Run coroutine when no event loop is active; skip otherwise."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    return None
