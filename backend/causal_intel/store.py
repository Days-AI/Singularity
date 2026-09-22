"""In-memory cache for enriched causal graphs."""
from __future__ import annotations

from collections import OrderedDict

from causal_intel.models import CausalIntelGraph

_MAX = 20
_store: OrderedDict[str, CausalIntelGraph] = OrderedDict()


def _key(session_id: str | None, flow_uuid: str | None) -> str | None:
    return session_id or flow_uuid


def put(graph: CausalIntelGraph) -> None:
    key = _key(graph.session_id, graph.flow_uuid)
    if not key:
        return
    _store[key] = graph
    _store.move_to_end(key)
    while len(_store) > _MAX:
        _store.popitem(last=False)


def get(session_id: str | None = None, flow_uuid: str | None = None) -> CausalIntelGraph | None:
    key = _key(session_id, flow_uuid)
    if not key:
        return None
    return _store.get(key)
