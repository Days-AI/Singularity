"""Optional LLM explanations for causal nodes."""
from __future__ import annotations

import logging

from causal_intel.models import CausalIntelGraph, NodeDetail

logger = logging.getLogger("singularity.causal_intel.explainer")


async def explain_node(detail: NodeDetail, graph: CausalIntelGraph) -> str:
    prompt = (
        f"Node: {detail.node.label}\n"
        f"Probability: {detail.node.probability}%\n"
        f"Question: {detail.node.question}\n"
        f"Incoming factors: {[e.source_label for e in detail.incoming]}\n"
        f"Outgoing factors: {[e.target_label for e in detail.outgoing]}\n"
        f"Goal: {graph.root_goal}\n"
        "Write 2 sentences explaining this node's causal role. Be concise."
    )
    try:
        from llm.ollama_client import get_ollama
        return await get_ollama().generate(
            "You are a causal analytics expert.",
            prompt,
            temperature=0.3,
            max_tokens=120,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Explainer fallback: %s", exc)
        return _template_explanation(detail, graph)


def _template_explanation(detail: NodeDetail, graph: CausalIntelGraph) -> str:
    inc = len(detail.incoming)
    out = len(detail.outgoing)
    return (
        f"{detail.node.label} registers {detail.node.probability:.0f}% probability "
        f"with {inc} upstream and {out} downstream causal links toward "
        f"\"{graph.root_goal[:60]}\"."
    )
