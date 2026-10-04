from benchmark.admission import run_benchmark


def test_admission_benchmark_matches_policy_contract():
    result = run_benchmark()

    assert result["cases"] == 6
    assert result["correct"] == 6
    assert result["policy_accuracy"] == 1.0
    assert result["silent_trust_leaks"] == 0
    assert result["silent_trust_rate"] == 0.0
