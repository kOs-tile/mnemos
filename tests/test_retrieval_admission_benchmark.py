import pytest

from benchmark.retrieval_admission import run_benchmark_async


def _scenario(result, label):
    return next(row for row in result["results"] if row["label"] == label)


@pytest.mark.asyncio
async def test_strict_policy_filters_direct_and_graph_candidates():
    result = await run_benchmark_async()
    strict = _scenario(result, "strict_evidence_sensitive")

    assert set(strict["returned_ids"]) == {
        "direct-safe-trace",
        "graph-safe-trace",
        "graph-safe-external",
    }
    assert strict["returned_count"] == 3
    assert strict["evaluated_candidates"] == 10
    assert strict["eligible_candidates"] == 3
    assert strict["policy_leak_count"] == 0
    assert strict["rejected_by_reason"] == {
        "stale_evidence": 2,
        "provenance_not_allowed": 2,
        "missing_source_trace": 2,
        "inactive_status": 1,
    }


@pytest.mark.asyncio
async def test_explicit_stale_opt_in_admits_only_policy_allowed_stale_memory():
    result = await run_benchmark_async()
    scenario = _scenario(result, "explicit_stale_opt_in")

    assert "direct-stale" in scenario["returned_ids"]
    assert "graph-stale" in scenario["returned_ids"]
    assert "direct-disallowed" not in scenario["returned_ids"]
    assert "graph-disallowed" not in scenario["returned_ids"]
    assert "direct-missing-trace" not in scenario["returned_ids"]
    assert "graph-missing-trace" not in scenario["returned_ids"]
    assert "graph-superseded" not in scenario["returned_ids"]
    assert scenario["policy_leak_count"] == 0


@pytest.mark.asyncio
async def test_default_advisory_remains_non_authority_but_less_restrictive():
    result = await run_benchmark_async()
    scenario = _scenario(result, "default_advisory")

    assert "direct-disallowed" in scenario["returned_ids"]
    assert "graph-disallowed" in scenario["returned_ids"]
    assert "direct-missing-trace" in scenario["returned_ids"]
    assert "graph-missing-trace" in scenario["returned_ids"]
    assert "direct-stale" not in scenario["returned_ids"]
    assert "graph-stale" not in scenario["returned_ids"]
    assert "graph-superseded" not in scenario["returned_ids"]
    assert scenario["policy_leak_count"] == 0


@pytest.mark.asyncio
async def test_retrieval_admission_corpus_has_zero_policy_leaks():
    result = await run_benchmark_async()

    assert result["scenarios"] == 3
    assert result["policy_leaks"] == 0
    for scenario in result["results"]:
        assert len(scenario["admission_fingerprint"]) == 64
