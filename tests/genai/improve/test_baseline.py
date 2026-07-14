"""Tests for statistical baseline helpers."""

from mlflow.genai.improve.trace_analyzer import _compute_baseline, _severity_from_z


def test_compute_baseline_normal_spike():
    values = [100.0, 110.0, 105.0, 50.0, 55.0, 48.0, 52.0, 51.0, 49.0, 53.0]
    bl = _compute_baseline(values)
    assert bl is not None
    assert bl["sufficient_data"] is True
    assert bl["z_score"] > 2.0
    assert bl["recent_mean"] > bl["baseline_mean"]


def test_compute_baseline_all_identical():
    bl = _compute_baseline([50.0] * 10)
    assert bl is not None
    assert bl["z_score"] == 0.0
    assert bl["stdev"] == 0.0


def test_compute_baseline_single_value():
    assert _compute_baseline([50.0]) is None


def test_compute_baseline_two_values():
    bl = _compute_baseline([100.0, 50.0])
    assert bl is not None
    assert bl["recent_mean"] == 100.0
    assert bl["baseline_mean"] == 50.0


def test_compute_baseline_insufficient_data():
    bl = _compute_baseline([10.0, 20.0, 30.0])
    assert bl is not None
    assert bl["sufficient_data"] is False


def test_compute_baseline_zero_variance_different_recent():
    bl = _compute_baseline([0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    assert bl is not None
    assert bl["z_score"] == -3.0


def test_compute_baseline_zero_variance_higher_recent():
    bl = _compute_baseline([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert bl is not None
    assert bl["z_score"] == 3.0


def test_compute_baseline_p95():
    values = list(range(1, 101))  # 1 to 100
    bl = _compute_baseline(values, recent_n=3, min_samples=5)
    assert bl is not None
    assert bl["p95"] >= 95


def test_severity_from_z_high():
    assert _severity_from_z(2.5) == "high"
    assert _severity_from_z(2.0) == "high"


def test_severity_from_z_medium():
    assert _severity_from_z(1.7) == "medium"
    assert _severity_from_z(1.5) == "medium"


def test_severity_from_z_low():
    assert _severity_from_z(1.2) == "low"
    assert _severity_from_z(1.0) == "low"


def test_severity_from_z_none():
    assert _severity_from_z(0.5) is None
    assert _severity_from_z(0.0) is None
    assert _severity_from_z(-1.0) is None


def test_severity_from_z_inverted():
    assert _severity_from_z(-2.0, invert=True) == "high"
    assert _severity_from_z(-1.5, invert=True) == "medium"
    assert _severity_from_z(-1.0, invert=True) == "low"
    assert _severity_from_z(-0.5, invert=True) is None


def test_severity_from_z_inverted_positive():
    assert _severity_from_z(2.0, invert=True) is None
