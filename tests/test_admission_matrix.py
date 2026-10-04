import pytest

from benchmark.admission_matrix import build_policy_cases, run_benchmark_async


def test_admission_policy_matrix_shape_is_stable():
    cases = build_policy_cases()

    assert len(cases) == 20
    assert sum(expected is None for _, _, _, expected in cases) == 6
    assert sum(expected is not None for _, _, _, expected in cases) == 14


@pytest.mark.asyncio
async def test_admission_policy_matrix_meets_v1_contract():
    result = await run_benchmark_async()

    assert result["policy_cases"] == 20
    assert result["policy_correct"] == 20
    assert result["policy_accuracy"] == 1.0
    assert result["policy_silent_trust_leaks"] == 0
    assert result["rejection_counts"] == {
        "provenance_not_allowed": 6,
        "stale_evidence": 3,
        "missing_source_trace": 2,
        "inactive_status": 3,
    }


@pytest.mark.asyncio
async def test_graph_expansion_preserves_same_trust_boundary():
    result = await run_benchmark_async()

    assert result["graph_cases"] == 4
    assert result["graph_correct"] == 4
    assert result["graph_policy_leaks"] == 0
    assert result["total_contract_checks"] == 24

    rows = {row["label"]: row for row in result["graph_results"]}
    assert rows["graph_stale_neighbor"]["actual_included"] is False
    assert rows["graph_disallowed_neighbor"]["actual_included"] is False
    assert rows["graph_missing_trace_neighbor"]["actual_included"] is False
    assert rows["graph_allowed_neighbor"]["actual_included"] is True
