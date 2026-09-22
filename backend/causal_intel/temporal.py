"""Timeline events from evidence and forecast."""
from __future__ import annotations

from causal_intel.models import CausalIntelGraph, TimelineEvent
from state import EvidenceItem


def build_timeline(
    graph: CausalIntelGraph,
    evidence: list[EvidenceItem],
) -> list[TimelineEvent]:
    events: list[TimelineEvent] = []
    for i, e in enumerate(evidence[:12]):
        events.append(
            TimelineEvent(
                date=f"t{i}",
                label=e.title[:48],
                kind="evidence",
                value=e.sentiment,
            )
        )
    for n in graph.nodes:
        if n.trend:
            for j, v in enumerate(n.trend[-8:]):
                events.append(
                    TimelineEvent(
                        date=f"f{j}",
                        label=n.label[:32],
                        kind="forecast",
                        value=v,
                    )
                )
    return events[:20]
