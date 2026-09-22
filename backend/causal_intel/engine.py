"""Graph traversal, scenario propagation, analytics."""
from __future__ import annotations

from collections import deque

from causal_intel.bayesian import apply_bayesian_posterior
from causal_intel.models import (
    AnalyticsBundle,
    CausalIntelGraph,
    EdgeSummary,
    InfluenceRank,
    IntelNode,
    LinkedEvidence,
    NodeDetail,
    PathDiscovery,
    ScenarioAssumption,
    ScenarioResult,
    SensitivityBar,
    TimelineEvent,
)
from causal_intel.temporal import build_timeline
from state import EvidenceItem


class CausalIntelEngine:
    def __init__(self, graph: CausalIntelGraph, evidence: list[EvidenceItem] | None = None) -> None:
        self.graph = graph
        self.evidence = evidence or []
        self._node_map = {n.id: n for n in graph.nodes}
        self._goal_id = next(
            (n.id for n in graph.nodes if n.node_type == "goal"),
            graph.nodes[0].id if graph.nodes else "goal_root",
        )

    @property
    def goal_id(self) -> str:
        return self._goal_id

    def get_node_detail(self, node_id: str, explanation: str | None = None) -> NodeDetail | None:
        node = self._node_map.get(node_id)
        if not node:
            return None

        incoming = [
            self._edge_summary(e)
            for e in self.graph.edges
            if e.target == node_id
        ]
        outgoing = [
            self._edge_summary(e)
            for e in self.graph.edges
            if e.source == node_id
        ]
        linked = [
            LinkedEvidence(
                index=i,
                source=e.source,
                title=e.title,
                detail=e.detail,
                sentiment=e.sentiment,
                url=e.url,
            )
            for i in node.evidence_ids
            if i < len(self.evidence)
            for e in [self.evidence[i]]
        ]
        fc = None
        if node.forecast_attached and node.trend:
            fc = {"trend": node.trend, "points": len(node.trend)}

        paths = self.discover_paths(node_id, self._goal_id, k=3).paths if node_id != self._goal_id else []

        return NodeDetail(
            node=node,
            incoming=incoming,
            outgoing=outgoing,
            evidence=linked,
            forecast=fc,
            monte_carlo=self.graph.monte_carlo,
            paths_to_goal=paths,
            explanation=explanation,
        )

    def _edge_summary(self, e: Any) -> EdgeSummary:
        src = self._node_map.get(e.source)
        tgt = self._node_map.get(e.target)
        return EdgeSummary(
            id=e.id,
            source_id=e.source,
            source_label=src.label if src else e.source,
            target_id=e.target,
            target_label=tgt.label if tgt else e.target,
            polarity=e.polarity,
            weight=e.weight,
            label=e.label,
        )

    def propagate_scenario(self, assumptions: list[ScenarioAssumption]) -> ScenarioResult:
        baseline = self.graph.overall_prediction
        posteriors: dict[str, float] = {n.id: n.probability for n in self.graph.nodes}

        disabled = {a.node_id for a in assumptions if not a.enabled}
        for a in assumptions:
            if not a.enabled:
                posteriors[a.node_id] = 50.0
            elif a.probability_override is not None:
                posteriors[a.node_id] = max(0.0, min(100.0, a.probability_override))

        posteriors = apply_bayesian_posterior(
            posteriors, self.graph.edges, assumptions, self._goal_id
        )

        goal_prob = self._compute_goal_from_posteriors(posteriors, disabled)
        deltas = {
            nid: round(posteriors.get(nid, 50.0) - self._node_map[nid].probability, 2)
            for nid in self._node_map
            if nid in posteriors
        }

        return ScenarioResult(
            goal_probability=round(goal_prob, 1),
            baseline_goal=round(baseline, 1),
            delta=round(goal_prob - baseline, 1),
            node_deltas=deltas,
            posterior_by_node={k: round(v, 1) for k, v in posteriors.items()},
        )

    def _compute_goal_from_posteriors(
        self, posteriors: dict[str, float], disabled: set[str]
    ) -> float:
        incoming = [e for e in self.graph.edges if e.target == self._goal_id]
        if not incoming:
            return posteriors.get(self._goal_id, self.graph.overall_prediction)

        contribs: list[float] = []
        weights: list[float] = []
        for e in incoming:
            if e.source in disabled:
                continue
            src_p = posteriors.get(e.source, 50.0)
            sign = 1.0 if e.polarity == "positive" else -1.0
            shifted = 50.0 + sign * (src_p - 50.0) * e.weight
            contribs.append(max(0.0, min(100.0, shifted)))
            weights.append(e.weight * e.confidence)

        if not contribs:
            return self.graph.overall_prediction

        total_w = sum(weights) or 1.0
        blended = sum(c * w for c, w in zip(contribs, weights)) / total_w
        base = posteriors.get(self._goal_id, self.graph.overall_prediction)
        return 0.6 * blended + 0.4 * base

    def rank_influences(self, limit: int = 8) -> list[InfluenceRank]:
        scores: dict[str, float] = {}
        polarity_map: dict[str, str | None] = {}
        for e in self.graph.edges:
            if e.target != self._goal_id and e.source not in scores:
                # accumulate path influence toward goal
                pass
            src = e.source
            tgt = e.target
            delta = e.weight * e.confidence
            if e.polarity == "negative":
                delta *= -1
            scores[src] = scores.get(src, 0.0) + abs(delta)
            polarity_map[src] = e.polarity

        ranked = sorted(scores.items(), key=lambda x: -x[1])[:limit]
        result: list[InfluenceRank] = []
        for nid, sc in ranked:
            node = self._node_map.get(nid)
            if not node:
                continue
            result.append(
                InfluenceRank(
                    node_id=nid,
                    label=node.label,
                    score=round(sc, 3),
                    polarity=polarity_map.get(nid),  # type: ignore[arg-type]
                )
            )
        return result

    def sensitivity(self, epsilon: float = 5.0) -> list[SensitivityBar]:
        baseline = self.graph.overall_prediction
        bars: list[SensitivityBar] = []
        for n in self.graph.nodes:
            if n.node_type == "goal":
                continue
            low_assump = [ScenarioAssumption(node_id=n.id, enabled=True, probability_override=max(0, n.probability - epsilon))]
            high_assump = [ScenarioAssumption(node_id=n.id, enabled=True, probability_override=min(100, n.probability + epsilon))]
            low = self.propagate_scenario(low_assump).goal_probability
            high = self.propagate_scenario(high_assump).goal_probability
            bars.append(
                SensitivityBar(
                    node_id=n.id,
                    label=n.label,
                    low=round(low, 1),
                    high=round(high, 1),
                    baseline=round(baseline, 1),
                )
            )
        return sorted(bars, key=lambda b: abs(b.high - b.low), reverse=True)

    def discover_paths(self, start_id: str, end_id: str, k: int = 5) -> PathDiscovery:
        paths: list[list[str]] = []
        labels: list[list[str]] = []
        queue: deque[tuple[str, list[str]]] = deque([(start_id, [start_id])])
        while queue and len(paths) < k:
            node, path = queue.popleft()
            if node == end_id:
                paths.append(path)
                labels.append([
                    self._node_map[n].label if n in self._node_map else n
                    for n in path
                ])
                continue
            if len(path) > 6:
                continue
            for e in self.graph.edges:
                if e.source == node and e.target not in path:
                    queue.append((e.target, path + [e.target]))
        return PathDiscovery(paths=paths, labels=labels)

    def analytics_bundle(self) -> AnalyticsBundle:
        return AnalyticsBundle(
            influence_ranking=self.rank_influences(),
            sensitivity=self.sensitivity(),
            timeline=build_timeline(self.graph, self.evidence),
            monte_carlo=self.graph.monte_carlo,
        )
