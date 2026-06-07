from __future__ import annotations

from agents import decision_engine
from fixtures import make_state, run_pipeline_metrics

_OPTION_TYPES = {"best", "alternative", "high_risk", "low_risk", "experimental"}


def test_decision_engine_five_options():
    state = run_pipeline_metrics(make_state())
    metrics = decision_engine.to_metrics(decision_engine.run(state))

    assert len(metrics["options"]) == 5
    types = {o["type"] for o in metrics["options"]}
    assert types == _OPTION_TYPES


def test_decision_engine_confidence_bounds():
    state = run_pipeline_metrics(make_state())
    for opt in decision_engine.to_metrics(decision_engine.run(state))["options"]:
        assert 0.0 <= opt["confidence"] <= 1.0
        assert opt["action"]
        assert opt["expected_outcome"]
        assert isinstance(opt["risks"], list)
