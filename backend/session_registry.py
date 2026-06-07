"""In-memory registry of completed simulation states.

The report endpoint needs the full `SingularityState` of a finished run to
re-synthesize a report without re-running the (expensive) pipeline. Supabase
persists the *outputs*, but not the rich intermediate state, so we keep the last
N completed states in process. This degrades gracefully: if a state has been
evicted (or the process restarted), the endpoint returns 404 and the UI prompts
the user to run a fresh simulation.
"""
from __future__ import annotations

from collections import OrderedDict

from state import SingularityState

# Cap retained states to bound memory; each holds up to ~1,500 persona records.
_MAX_SESSIONS = 20
_store: "OrderedDict[str, SingularityState]" = OrderedDict()


def put(session_id: str, state: SingularityState) -> None:
    """Cache a completed run's state, evicting the oldest beyond the cap."""
    _store[session_id] = state
    _store.move_to_end(session_id)
    while len(_store) > _MAX_SESSIONS:
        _store.popitem(last=False)


def get(session_id: str) -> SingularityState | None:
    return _store.get(session_id)
