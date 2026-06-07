from __future__ import annotations

from agents import consensus as consensus_agent
from tests.fixtures import make_state, run_pipeline_metrics


def test_consensus_merge_weights():
    state = run_pipeline_metrics(make_state())
    state.metrics["deliberation"] = {
        "agreement_rate": 0.7,
        "polarization_index": 0.15,
        "mean_sentiment": 0.2,
        "narrative_clusters": [],
    }
    state.metrics["social_simulation"] = {
        "contagion_index": 0.4,
        "polarization_index": 0.18,
    }
    state.metrics["council"] = {
        "synthesis": "Launch with trust-building creative.",
        "opinions": [
            {"specialist_id": "pr", "confidence": 0.8, "recommendation": "Proceed with launch"},
            {"specialist_id": "brand", "confidence": 0.75, "recommendation": "Invest in brand trust"},
        ],
    }

    metrics, payload = consensus_agent.run(state)

    assert 0 <= payload.agreement_score <= 1
    assert payload.recommended_action
    assert metrics["agreement_score"] == payload.agreement_score
    assert isinstance(payload.supporting_signals, list)
    assert 0 <= payload.council_alignment <= 1


def test_consensus_dissent_detection():
    state = make_state()
    state.metrics["deliberation"] = {
        "agreement_rate": 0.4,
        "polarization_index": 0.5,
        "narrative_clusters": [{"label": "A"}, {"label": "B"}, {"label": "C"}],
    }
    state.metrics["council"] = {
        "opinions": [{"confidence": 0.3, "recommendation": "wait"}],
    }
    state.metrics["prediction_market"] = {"overall_outcome": 45}

    _, payload = consensus_agent.run(state)
    assert "polarized" in payload.dissent.lower() or "narratives" in payload.dissent.lower()
