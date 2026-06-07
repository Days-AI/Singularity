from __future__ import annotations

import inspect


def test_flow_imports_social_council_consensus():
    import flow  # noqa: F401

    from agents import consensus, council, social

    assert callable(social.run)
    assert callable(council.run)
    assert callable(consensus.run)


def test_flow_phase_order():
    import flow

    source = inspect.getsource(flow.SingularityFlow.run)
    psych_idx = source.index("_run_phase(\"psychometric\"")
    social_idx = source.index("_run_phase(\"social_simulation\"")
    parallel_idx = source.index("_run_phase(\"parallel_post_social\"")
    forecast_idx = source.index("_run_phase(\"forecast\"")
    consensus_idx = source.index("_run_phase(\"consensus_engine\"")
    decision_idx = source.index("_run_phase(\"decision_engine\"")
    report_idx = source.index("_run_phase(\"report\"")

    assert psych_idx < social_idx < parallel_idx < forecast_idx
    assert consensus_idx < decision_idx < report_idx
    assert "rag_index" not in source.split("_run_phase(\"report\"")[1].split("_finish")[0]


def test_flow_has_parallel_post_social():
    import flow

    sf = flow.SingularityFlow
    assert hasattr(sf, "_run_parallel_post_social")
    assert hasattr(sf, "_run_analytics_bundle")
