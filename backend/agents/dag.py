"""DAG decomposer (spec PT-01).

Calls local Gemma to turn an unstructured query into a JSON dependency graph of
atomic base_roots, validates/repairs it into typed models, and derives edges.
Falls back to a sensible default DAG if the model is unavailable or returns
unusable output, so the flow always proceeds.
"""
from __future__ import annotations

import logging

from config import get_settings
from llm.ollama_client import get_ollama
from prompts import DAG_SYSTEM, DAG_USER
from state import DagCreatedPayload, DagEdge, DagNode

logger = logging.getLogger("singularity.dag")

_VALID_TYPES = {"web_search", "financial", "psychometric", "forecast"}


async def decompose(query: str) -> DagCreatedPayload:
    try:
        data = await get_ollama().generate_json(
            DAG_SYSTEM, DAG_USER.format(query=query),
            max_tokens=get_settings().dag_max_tokens,
        )
        nodes = _coerce_nodes(data.get("base_roots", []))
        if not nodes:
            raise ValueError("model returned no usable base_roots")
    except Exception as exc:  # noqa: BLE001
        logger.warning("DAG decomposition fell back to default: %s", exc)
        nodes = _default_nodes()

    nodes = _ensure_coverage(nodes)
    _ensure_dependencies(nodes)
    edges = _edges_from(nodes)
    return DagCreatedPayload(root_query=query, nodes=nodes, edges=edges)


def _coerce_nodes(raw: list) -> list[DagNode]:
    nodes: list[DagNode] = []
    seen: set[str] = set()
    for i, item in enumerate(raw[:8]):
        if not isinstance(item, dict):
            continue
        agent_type = str(item.get("agent_type", "")).strip()
        if agent_type not in _VALID_TYPES:
            continue
        node_id = str(item.get("id") or f"br_{i + 1:03d}")
        if node_id in seen:
            node_id = f"{node_id}_{i}"
        seen.add(node_id)
        deps = [str(d) for d in item.get("dependencies", []) if isinstance(d, (str, int))]
        nodes.append(
            DagNode(
                id=node_id,
                task=str(item.get("task", "Analyze sub-problem")).strip()[:160],
                agent_type=agent_type,  # type: ignore[arg-type]
                dependencies=deps,
                priority=int(item.get("priority", 1) or 1),
            )
        )
    # Drop dependency references to ids that don't exist (prevents broken edges).
    valid_ids = {n.id for n in nodes}
    for n in nodes:
        n.dependencies = [d for d in n.dependencies if d in valid_ids and d != n.id]
    return nodes


def _ensure_coverage(nodes: list[DagNode]) -> list[DagNode]:
    """Guarantee the pipeline has psychometric + forecast stages."""
    types = {n.agent_type for n in nodes}
    evidence_ids = [n.id for n in nodes if n.agent_type in ("web_search", "financial")]
    if "psychometric" not in types:
        nodes.append(
            DagNode(
                id="br_psy",
                task="Psychometric segment responses - 1500 IPIP-300 personas",
                agent_type="psychometric",
                dependencies=evidence_ids,
                priority=2,
            )
        )
    if "forecast" not in types:
        psy_ids = [n.id for n in nodes if n.agent_type == "psychometric"]
        nodes.append(
            DagNode(
                id="br_fct",
                task="Time-series projection (90-day horizon)",
                agent_type="forecast",
                dependencies=evidence_ids + psy_ids,
                priority=3,
            )
        )
    return nodes


def _ensure_dependencies(nodes: list[DagNode]) -> None:
    """If the model emitted a flat list (no deps), wire a sensible pipeline:
    evidence roots -> psychometric -> forecast, so the DAG renders connected."""
    if any(n.dependencies for n in nodes):
        return
    evidence_ids = [n.id for n in nodes if n.agent_type in ("web_search", "financial")]
    psy_ids = [n.id for n in nodes if n.agent_type == "psychometric"]
    for n in nodes:
        if n.agent_type == "psychometric":
            n.dependencies = list(evidence_ids)
        elif n.agent_type == "forecast":
            n.dependencies = list(evidence_ids) + list(psy_ids)


def _edges_from(nodes: list[DagNode]) -> list[DagEdge]:
    return [DagEdge(source=dep, target=n.id) for n in nodes for dep in n.dependencies]


def _default_nodes() -> list[DagNode]:
    return [
        DagNode(id="br_001", task="Audience and competitive landscape - web research",
                agent_type="web_search", dependencies=[], priority=1),
        DagNode(id="br_002", task="Brand & message resonance - public discourse and media",
                agent_type="web_search", dependencies=[], priority=1),
        DagNode(id="br_003", task="Audience sentiment & public discourse",
                agent_type="web_search", dependencies=[], priority=1),
        DagNode(id="br_004", task="Psychometric segment responses - 1500 IPIP-300 personas",
                agent_type="psychometric", dependencies=["br_002", "br_003"], priority=2),
        DagNode(id="br_005", task="Time-series projection (90-day horizon)",
                agent_type="forecast", dependencies=["br_001", "br_004"], priority=3),
    ]
