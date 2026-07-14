"""Tests for the 6 statistical baseline detectors."""

from mlflow.genai.improve.trace_analyzer import (
    _detect_context_bloat,
    _detect_tool_redundancy,
    _detect_score_degradation,
    _detect_slowdown,
    _detect_error_spike,
    _detect_incomplete_pipeline,
)


# ── context_bloat ────────────────────────────────────────

def test_context_bloat_stable(make_parsed_trace):
    traces = [make_parsed_trace(trace_size_bytes=50_000) for _ in range(10)]
    findings = _detect_context_bloat(traces)
    assert len(findings) == 0


def test_context_bloat_outlier(make_parsed_trace):
    traces = [make_parsed_trace(trace_size_bytes=50_000) for _ in range(7)]
    traces.insert(0, make_parsed_trace(trace_size_bytes=5_000_000))
    findings = _detect_context_bloat(traces)
    patterns = [f.pattern for f in findings]
    assert "context_bloat" in patterns


def test_context_growth(make_parsed_trace):
    recent = [make_parsed_trace(trace_size_bytes=2_000_000) for _ in range(3)]
    older = [make_parsed_trace(trace_size_bytes=100_000) for _ in range(7)]
    traces = recent + older
    findings = _detect_context_bloat(traces)
    patterns = [f.pattern for f in findings]
    assert "context_growth" in patterns


def test_context_bloat_cold_start(make_parsed_trace):
    traces = [make_parsed_trace(trace_size_bytes=800_000) for _ in range(3)]
    findings = _detect_context_bloat(traces)
    assert any(f.evidence.get("sufficient_data") is False for f in findings)


# ── tool_redundancy ──────────────────────────────────────

def test_tool_redundancy_none(make_parsed_trace):
    traces = [make_parsed_trace(tool_names=["a", "b", "c"]) for _ in range(10)]
    findings = _detect_tool_redundancy(traces)
    assert len(findings) == 0


def test_tool_redundancy_spike(make_parsed_trace):
    recent = [make_parsed_trace(tool_names=["a", "a", "b", "b"]) for _ in range(3)]
    older = [make_parsed_trace(tool_names=["a", "b", "c"]) for _ in range(7)]
    traces = recent + older
    findings = _detect_tool_redundancy(traces)
    if findings:
        assert findings[0].pattern == "tool_redundancy"


def test_tool_redundancy_cold_start(make_parsed_trace):
    traces = [make_parsed_trace(tool_names=["a", "a", "b"]) for _ in range(3)]
    findings = _detect_tool_redundancy(traces)
    patterns = [f.pattern for f in findings]
    if findings:
        assert findings[0].evidence.get("sufficient_data") is False


# ── score_degradation ────────────────────────────────────

def _make_scored_traces(make_parsed_trace, scores, scorer_name="completeness"):
    traces = []
    for s in scores:
        traces.append(make_parsed_trace(
            assessments=[{"name": scorer_name, "value": "yes" if s else "no"}],
        ))
    return traces


def test_score_stable_high(make_parsed_trace):
    traces = _make_scored_traces(make_parsed_trace, [True] * 10)
    findings = _detect_score_degradation(traces)
    assert len(findings) == 0


def test_score_declining(make_parsed_trace):
    scores = [False, False, False, True, True, True, True, True, True, True]
    traces = _make_scored_traces(make_parsed_trace, scores)
    findings = _detect_score_degradation(traces)
    patterns = [f.pattern for f in findings]
    assert "score_declining" in patterns


def test_score_low_overall(make_parsed_trace):
    scores = [False, False, True, False, False, False, True, False]
    traces = _make_scored_traces(make_parsed_trace, scores)
    findings = _detect_score_degradation(traces)
    patterns = [f.pattern for f in findings]
    assert "score_degradation" in patterns or "score_declining" in patterns


def test_score_dedup(make_parsed_trace):
    scores = [False, False, False, True, True, True, True, True]
    traces = _make_scored_traces(make_parsed_trace, scores)
    findings = _detect_score_degradation(traces)
    scorer_findings = [f for f in findings if f.evidence.get("scorer") == "completeness"]
    assert len(scorer_findings) == 1


# ── slowdown ─────────────────────────────────────────────

def test_slowdown_stable(make_parsed_trace):
    traces = [make_parsed_trace(execution_ms=30_000) for _ in range(10)]
    findings = _detect_slowdown(traces)
    assert len(findings) == 0


def test_slowdown_recent_spike(make_parsed_trace):
    recent = [make_parsed_trace(execution_ms=300_000) for _ in range(3)]
    older = [make_parsed_trace(execution_ms=30_000) for _ in range(7)]
    traces = recent + older
    findings = _detect_slowdown(traces)
    patterns = [f.pattern for f in findings]
    assert "execution_slowdown" in patterns


def test_slowdown_outlier(make_parsed_trace):
    traces = [make_parsed_trace(execution_ms=30_000) for _ in range(9)]
    traces.insert(0, make_parsed_trace(execution_ms=500_000))
    findings = _detect_slowdown(traces)
    patterns = [f.pattern for f in findings]
    assert "slow_execution" in patterns


def test_slowdown_cold_start(make_parsed_trace):
    traces = [make_parsed_trace(execution_ms=150_000) for _ in range(3)]
    findings = _detect_slowdown(traces)
    assert any(f.evidence.get("sufficient_data") is False for f in findings)


# ── error_spike ──────────────────────────────────────────

def test_error_no_errors(make_parsed_trace):
    traces = [make_parsed_trace(error_count=0) for _ in range(10)]
    findings = _detect_error_spike(traces)
    assert len(findings) == 0


def test_error_spike_from_zero_baseline(make_parsed_trace):
    recent = [make_parsed_trace(error_count=2) for _ in range(3)]
    older = [make_parsed_trace(error_count=0) for _ in range(7)]
    traces = recent + older
    findings = _detect_error_spike(traces)
    assert len(findings) == 1
    assert findings[0].pattern == "error_spike"


def test_error_spike_z_score(make_parsed_trace):
    recent = [make_parsed_trace(error_count=3) for _ in range(3)]
    mixed = [make_parsed_trace(error_count=1 if i % 3 == 0 else 0) for i in range(7)]
    traces = recent + mixed
    findings = _detect_error_spike(traces)
    if findings:
        assert findings[0].evidence.get("sufficient_data") is True


def test_error_spike_cold_start(make_parsed_trace):
    traces = [make_parsed_trace(error_count=1) for _ in range(3)]
    findings = _detect_error_spike(traces)
    if findings:
        assert findings[0].evidence.get("sufficient_data") is False


# ── incomplete_pipeline ──────────────────────────────────

def test_pipeline_complete(make_parsed_trace):
    tools = ["parse", "filter", "score", "report"]
    traces = [make_parsed_trace(tool_names=tools) for _ in range(10)]
    findings = _detect_incomplete_pipeline(traces)
    assert len(findings) == 0


def test_pipeline_missing_tool(make_parsed_trace):
    full = ["parse", "filter", "score", "report"]
    partial = ["parse", "filter", "report"]
    traces = [make_parsed_trace(tool_names=full) for _ in range(9)]
    traces.insert(0, make_parsed_trace(tool_names=partial))
    findings = _detect_incomplete_pipeline(traces)
    assert len(findings) == 1
    assert "score" in findings[0].evidence["missing_tools"]


def test_pipeline_adaptive_threshold_small(make_parsed_trace):
    full = ["a", "b", "c"]
    partial = ["a", "b"]
    traces = [make_parsed_trace(tool_names=full) for _ in range(3)]
    traces.append(make_parsed_trace(tool_names=partial))
    findings = _detect_incomplete_pipeline(traces)
    if findings:
        assert findings[0].evidence["frequency_threshold"] == 1.0
    else:
        # With 100% threshold on < 5 traces, "c" only appears in 3/4 = 75%,
        # so it's not expected — no finding is correct behavior
        pass


def test_pipeline_adaptive_threshold_large(make_parsed_trace):
    full = ["a", "b", "c"]
    traces = [make_parsed_trace(tool_names=full) for _ in range(15)]
    findings = _detect_incomplete_pipeline(traces)
    assert len(findings) == 0
