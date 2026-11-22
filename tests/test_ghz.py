"""GHZ correctness tests (CPU target)."""
import cudaq
import pytest

from ghz_benchmark import check_ghz_correctness, ghz, speedups


@pytest.fixture(autouse=True)
def cpu_target():
    cudaq.set_target("qpp-cpu")


def test_ghz_only_all_zero_and_all_one():
    counts = check_ghz_correctness(n=8, shots=4000)
    assert set(counts) == {"0" * 8, "1" * 8}


def test_ghz_balanced_at_other_sizes():
    for n in (3, 12):
        counts = dict(cudaq.sample(ghz, n, shots_count=3000).items())
        assert set(counts) <= {"0" * n, "1" * n}
        assert abs(counts.get("0" * n, 0) / 3000 - 0.5) < 0.06


def test_speedup_pairs_only_common_sizes():
    rows = [
        {"target": "qpp-cpu", "n": 20, "median_s": 4.0, "std_s": 0, "est_mem_gb": 0},
        {"target": "qpp-cpu", "n": 21, "median_s": 8.0, "std_s": 0, "est_mem_gb": 0},
        {"target": "nvidia", "n": 20, "median_s": 0.5, "std_s": 0, "est_mem_gb": 0},
    ]
    assert speedups(rows) == [(20, 8.0)]
