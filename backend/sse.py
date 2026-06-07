"""SSE plumbing: a per-flow async event bus plus wire framing.

The flow pushes typed events onto an `EventBus`; the streaming endpoint drains
it and writes properly framed Server-Sent Events. Each frame carries an
incrementing `id:` so the browser EventSource can resume via Last-Event-ID.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

# Sentinel pushed by the flow to signal the stream may close.
STREAM_DONE = object()


@dataclass
class Event:
    type: str
    payload: dict[str, Any]


class EventBus:
    """A single-consumer async queue of SSE events for one flow run."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        self._closed = False

    async def emit(self, event_type: str, payload: BaseModel | dict[str, Any]) -> None:
        if self._closed:
            return
        data = payload.model_dump() if isinstance(payload, BaseModel) else payload
        await self._queue.put(Event(type=event_type, payload=data))

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._queue.put(STREAM_DONE)

    async def drain(self):
        """Yield events until the flow closes the bus."""
        while True:
            item = await self._queue.get()
            if item is STREAM_DONE:
                return
            yield item


def frame(event_id: int, event_type: str, payload: dict[str, Any]) -> str:
    """Format one SSE frame. `id:` enables Last-Event-ID resume on the client."""
    data = json.dumps(payload, separators=(",", ":"), default=str)
    return f"id: {event_id}\nevent: {event_type}\ndata: {data}\n\n"
