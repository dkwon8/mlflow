"""
Trace analyzer for the MLflow improve system.

Reads raw trace spans and assessments from MLflow to detect patterns
that indicate quality degradation, inefficiency, or scaling issues.
Works universally with any MLflow-traced agent — no custom tags required.

Each detection function returns a list of findings that the suggestion
engine uses to generate actionable fixes.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from statistics import mean as _mean, pstdev as _pstdev


HEAL_PATTERNS = {"error_spike"}
IMPROVE_PATTERNS = {
    "context_bloat", "context_growth", "tool_redundancy",
    "score_degradation", "score_declining",
    "slow_execution", "execution_slowdown", "incomplete_pipeline",
}


@dataclass
class Finding:
    """A detected issue from trace analysis."""
    pattern: str
    severity: str  # "low", "medium", "high"
    description: str
    category: str = "improve"  # "heal" or "improve"
    evidence: dict = field(default_factory=dict)


def _compute_baseline(
    values: list[float],
    recent_n: int = 3,
    min_samples: int = 5,
) -> dict | None:
    """Compute statistical baseline from a series of metric values.

    Splits values into a recent window (first recent_n) and a baseline
    (the rest), then computes how far the recent window deviates from
    the baseline in standard deviations (z-score).

    Returns None if fewer than 2 values are available.
    """
    if len(values) < 2:
        return None

    recent_n = min(recent_n, len(values) - 1)
    recent = values[:recent_n]
    baseline = values[recent_n:]

    overall_mean = _mean(values)
    overall_stdev = _pstdev(values)

    sorted_vals = sorted(values)
    p95_idx = min(int(len(sorted_vals) * 0.95), len(sorted_vals) - 1)

    recent_mean = _mean(recent)
    baseline_mean = _mean(baseline)
    baseline_stdev = _pstdev(baseline)

    if baseline_stdev > 0:
        z_score = (recent_mean - baseline_mean) / baseline_stdev
    elif recent_mean != baseline_mean:
        z_score = 3.0 if recent_mean > baseline_mean else -3.0
    else:
        z_score = 0.0

    return {
        "mean": overall_mean,
        "stdev": overall_stdev,
        "p95": sorted_vals[p95_idx],
        "recent_mean": recent_mean,
        "baseline_mean": baseline_mean,
        "baseline_stdev": baseline_stdev,
        "z_score": z_score,
        "sufficient_data": len(values) >= min_samples,
    }


def _severity_from_z(z: float, invert: bool = False) -> str | None:
    """Map a z-score to a severity level.

    For metrics where higher is worse (latency, errors), use default.
    For metrics where lower is worse (pass rates), set invert=True.
    """
    effective_z = -z if invert else z
    if effective_z >= 2.0:
        return "high"
    if effective_z >= 1.5:
        return "medium"
    if effective_z >= 1.0:
        return "low"
    return None


def _extract_span_info(span: dict) -> dict:
    """Extract useful fields from a raw span dict."""
    attrs = span.get("attributes", {})
    if isinstance(attrs, str):
        try:
            import ast
            attrs = ast.literal_eval(attrs)
        except (ValueError, SyntaxError):
            attrs = {}

    status = span.get("status", {})
    if isinstance(status, str):
        try:
            import ast
            status = ast.literal_eval(status)
        except (ValueError, SyntaxError):
            status = {}

    span_type = attrs.get("mlflow.spanType", "").strip('"')
    status_code = status.get("code", "") if isinstance(status, dict) else ""
    status_message = status.get("message", "") if isinstance(status, dict) else ""

    span_inputs = attrs.get("mlflow.spanInputs", "")
    if isinstance(span_inputs, str):
        span_inputs = span_inputs.strip('"')

    return {
        "name": span.get("name", ""),
        "span_type": span_type,
        "is_error": status_code == "STATUS_CODE_ERROR",
        "error_message": status_message if status_code == "STATUS_CODE_ERROR" else "",
        "start_ns": int(span.get("start_time_unix_nano", 0)),
        "end_ns": int(span.get("end_time_unix_nano", 0)),
        "inputs": span_inputs,
        "parent_span_id": span.get("parent_span_id"),
    }


def _parse_trace(trace_row: dict) -> dict:
    """Parse a raw trace row into a structured dict for analysis.

    Extracts tool calls, errors, execution time, trace size, and
    assessments directly from span data — no custom tags needed.
    """
    spans = trace_row.get("spans", [])
    if not spans:
        spans = []

    tool_names = []
    error_count = 0
    error_details = []
    user_query = ""
    trace_start_ns = 0

    for span in spans:
        info = _extract_span_info(span)
        if info["span_type"] == "TOOL":
            tool_names.append(info["name"])
        if info["is_error"]:
            error_count += 1
            error_details.append({
                "span_name": info["name"],
                "error_message": info["error_message"],
            })
        if info["parent_span_id"] in (None, "None", ""):
            if info["start_ns"] > 0:
                trace_start_ns = info["start_ns"]
        if info["parent_span_id"] in (None, "None", "") and info["inputs"]:
            try:
                inputs = json.loads(info["inputs"]) if isinstance(info["inputs"], str) else info["inputs"]
                if isinstance(inputs, list) and inputs:
                    first = inputs[0]
                    user_query = first.get("content", str(first)) if isinstance(first, dict) else str(first)
                elif isinstance(inputs, str):
                    user_query = inputs
            except (json.JSONDecodeError, TypeError):
                user_query = str(info["inputs"])[:200]

    tool_counts = Counter(tool_names)
    unique_tools = set(tool_names)
    duplicates = {name: count for name, count in tool_counts.items() if count > 1}

    trace_size = len(json.dumps(spans, default=str).encode())
    execution_ms = int(trace_row.get("execution_duration", 0) or 0)

    assessments = []
    for a in trace_row.get("assessments", []):
        if isinstance(a, dict):
            name = a.get("assessment_name", a.get("name", ""))
            feedback = a.get("feedback", {})
            value = feedback.get("value") if isinstance(feedback, dict) else None
            if value is None:
                value = a.get("value", a.get("string_value"))
            assessments.append({"name": name, "value": value})

    return {
        "trace_id": trace_row.get("trace_id", ""),
        "tool_names": tool_names,
        "unique_tools": unique_tools,
        "duplicate_tools": duplicates,
        "tool_call_count": len(tool_names),
        "error_count": error_count,
        "error_details": error_details,
        "user_query": user_query,
        "execution_ms": execution_ms,
        "trace_size_bytes": trace_size,
        "assessments": assessments,
        "start_ns": trace_start_ns,
    }


def analyze_traces(traces_data: list[dict], mode: str | None = None) -> list[Finding]:
    """Run all detection patterns against a set of traces.

    Args:
        traces_data: List of raw trace dicts from mlflow.search_traces().
            Each dict should have keys: spans, execution_duration, assessments.
        mode: Optional "heal" or "improve" to filter findings by category.

    Returns:
        List of Finding objects describing detected issues.
    """
    if not traces_data:
        return []

    parsed = [_parse_trace(t) for t in traces_data]

    findings = []
    findings.extend(_detect_context_bloat(parsed))
    findings.extend(_detect_tool_redundancy(parsed))
    findings.extend(_detect_score_degradation(parsed))
    findings.extend(_detect_slowdown(parsed))
    findings.extend(_detect_error_spike(parsed))
    findings.extend(_detect_incomplete_pipeline(parsed))

    if mode:
        target = HEAL_PATTERNS if mode == "heal" else IMPROVE_PATTERNS
        findings = [f for f in findings if f.pattern in target]

    return findings


def _detect_context_bloat(traces: list[dict]) -> list[Finding]:
    """Detect if trace sizes are abnormally large or growing."""
    sizes = [t["trace_size_bytes"] for t in traces if t["trace_size_bytes"] > 0]

    if len(sizes) < 2:
        return []

    findings = []
    bl = _compute_baseline(sizes)
    max_size = max(sizes)

    if bl and bl["sufficient_data"]:
        max_z = (max_size - bl["mean"]) / bl["stdev"] if bl["stdev"] > 0 else 0.0
        severity = _severity_from_z(max_z)
        if severity:
            findings.append(Finding(
                pattern="context_bloat",
                severity=severity,
                description=f"Trace sizes averaging {bl['mean'] / 1_000_000:.1f}MB, max {max_size / 1_000_000:.1f}MB ({max_z:.1f}σ above mean).",
                evidence={
                    "avg_size_bytes": int(bl["mean"]),
                    "max_size_bytes": max_size,
                    "baseline_mean": int(bl["baseline_mean"]),
                    "baseline_stdev": int(bl["baseline_stdev"]),
                    "z_score": round(max_z, 2),
                    "sufficient_data": True,
                },
            ))
    elif bl and bl["p95"] > 500_000:
        findings.append(Finding(
            pattern="context_bloat",
            severity="medium" if max_size > 1_000_000 else "low",
            description=f"Trace sizes p95 at {bl['p95'] / 1_000_000:.1f}MB (limited data, {len(sizes)} traces).",
            evidence={
                "avg_size_bytes": int(bl["mean"]),
                "max_size_bytes": max_size,
                "p95": int(bl["p95"]),
                "sufficient_data": False,
            },
        ))

    if bl and bl["sufficient_data"]:
        growth_severity = _severity_from_z(bl["z_score"])
        if growth_severity:
            findings.append(Finding(
                pattern="context_growth",
                severity=growth_severity,
                description=f"Trace sizes growing — recent {bl['recent_mean'] / 1_000_000:.1f}MB vs baseline {bl['baseline_mean'] / 1_000_000:.1f}MB ({bl['z_score']:.1f}σ).",
                evidence={
                    "recent_avg": int(bl["recent_mean"]),
                    "baseline_mean": int(bl["baseline_mean"]),
                    "baseline_stdev": int(bl["baseline_stdev"]),
                    "z_score": round(bl["z_score"], 2),
                    "sufficient_data": True,
                },
            ))

    return findings


def _detect_tool_redundancy(traces: list[dict]) -> list[Finding]:
    """Detect if the agent is making redundant tool calls by analyzing span data."""
    duplicate_counts: dict[str, int] = {}
    indicators = []

    for t in traces:
        has_dupes = bool(t["duplicate_tools"])
        indicators.append(1.0 if has_dupes else 0.0)
        if has_dupes:
            for tool_name in t["duplicate_tools"]:
                duplicate_counts[tool_name] = duplicate_counts.get(tool_name, 0) + 1

    if not duplicate_counts:
        return []

    findings = []
    bl = _compute_baseline(indicators)
    dupe_rate = sum(indicators) / len(indicators)
    worst_tool = max(duplicate_counts, key=duplicate_counts.get)

    if bl and bl["sufficient_data"]:
        severity = _severity_from_z(bl["z_score"])
        if severity:
            findings.append(Finding(
                pattern="tool_redundancy",
                severity=severity,
                description=f"Tool redundancy in {int(sum(indicators))}/{len(traces)} traces ({dupe_rate:.0%}, {bl['z_score']:.1f}σ above baseline). Most duplicated: {worst_tool}.",
                evidence={
                    "duplicate_counts": duplicate_counts,
                    "traces_affected": int(sum(indicators)),
                    "rate": round(dupe_rate, 2),
                    "baseline_mean": round(bl["baseline_mean"], 2),
                    "baseline_stdev": round(bl["baseline_stdev"], 2),
                    "z_score": round(bl["z_score"], 2),
                    "sufficient_data": True,
                },
            ))
    elif dupe_rate > 0.5:
        findings.append(Finding(
            pattern="tool_redundancy",
            severity="medium",
            description=f"Tool redundancy in {int(sum(indicators))}/{len(traces)} traces ({dupe_rate:.0%}, limited data). Most duplicated: {worst_tool}.",
            evidence={
                "duplicate_counts": duplicate_counts,
                "traces_affected": int(sum(indicators)),
                "rate": round(dupe_rate, 2),
                "sufficient_data": False,
            },
        ))

    return findings


def _detect_score_degradation(traces: list[dict]) -> list[Finding]:
    """Detect if assessment scores are low or trending downward."""
    score_map: dict[str, list[float]] = {}

    for t in traces:
        for a in t["assessments"]:
            name = a.get("name", "")
            if not name:
                continue
            val = a.get("value")
            if val in ("yes", "true", "True", True):
                score_map.setdefault(name, []).append(1.0)
            elif val in ("no", "false", "False", False):
                score_map.setdefault(name, []).append(0.0)

    findings = []
    declined_scorers = set()

    for name, scores in score_map.items():
        if len(scores) < 3:
            continue

        pass_rate = _mean(scores)
        bl = _compute_baseline(scores)

        if bl and bl["sufficient_data"]:
            decline_severity = _severity_from_z(bl["z_score"], invert=True)
            if decline_severity:
                declined_scorers.add(name)
                findings.append(Finding(
                    pattern="score_declining",
                    severity=decline_severity,
                    description=f"{name} declining — recent {bl['recent_mean']:.0%} vs baseline {bl['baseline_mean']:.0%} ({abs(bl['z_score']):.1f}σ drop).",
                    evidence={
                        "scorer": name,
                        "recent_rate": round(bl["recent_mean"], 2),
                        "baseline_mean": round(bl["baseline_mean"], 2),
                        "baseline_stdev": round(bl["baseline_stdev"], 2),
                        "z_score": round(bl["z_score"], 2),
                        "sufficient_data": True,
                    },
                ))

        if pass_rate < 0.5 and name not in declined_scorers:
            findings.append(Finding(
                pattern="score_degradation",
                severity="high" if pass_rate < 0.3 else "medium",
                description=f"{name} passing only {pass_rate:.0%} of the time ({int(sum(scores))}/{len(scores)} traces).",
                evidence={
                    "scorer": name,
                    "pass_rate": round(pass_rate, 2),
                    "total_traces": len(scores),
                    "baseline_mean": round(bl["baseline_mean"], 2) if bl else None,
                    "sufficient_data": bl["sufficient_data"] if bl else False,
                },
            ))

    return findings


def _detect_slowdown(traces: list[dict]) -> list[Finding]:
    """Detect if execution time is abnormally high or increasing."""
    times = [float(t["execution_ms"]) for t in traces if t["execution_ms"] > 0]

    if len(times) < 3:
        return []

    findings = []
    bl = _compute_baseline(times)

    if bl and bl["sufficient_data"]:
        max_time = max(times)
        max_z = (max_time - bl["mean"]) / bl["stdev"] if bl["stdev"] > 0 else 0.0
        severity = _severity_from_z(max_z)
        if severity:
            findings.append(Finding(
                pattern="slow_execution",
                severity=severity,
                description=f"Execution time p95 at {bl['p95'] / 1000:.0f}s, max {max_time / 1000:.0f}s ({max_z:.1f}σ above mean of {bl['mean'] / 1000:.0f}s).",
                evidence={
                    "avg_ms": int(bl["mean"]),
                    "p95_ms": int(bl["p95"]),
                    "max_ms": int(max_time),
                    "baseline_mean": int(bl["baseline_mean"]),
                    "baseline_stdev": int(bl["baseline_stdev"]),
                    "z_score": round(max_z, 2),
                    "sufficient_data": True,
                },
            ))

        slowdown_severity = _severity_from_z(bl["z_score"])
        if slowdown_severity:
            findings.append(Finding(
                pattern="execution_slowdown",
                severity=slowdown_severity,
                description=f"Execution slowing — recent {bl['recent_mean'] / 1000:.0f}s vs baseline {bl['baseline_mean'] / 1000:.0f}s ({bl['z_score']:.1f}σ).",
                evidence={
                    "recent_avg_ms": int(bl["recent_mean"]),
                    "baseline_mean_ms": int(bl["baseline_mean"]),
                    "baseline_stdev_ms": int(bl["baseline_stdev"]),
                    "z_score": round(bl["z_score"], 2),
                    "sufficient_data": True,
                },
            ))
    elif bl:
        avg_time = bl["mean"]
        if avg_time > 120_000:
            findings.append(Finding(
                pattern="slow_execution",
                severity="medium" if avg_time > 180_000 else "low",
                description=f"Average execution time is {avg_time / 1000:.0f}s (limited data, {len(times)} traces).",
                evidence={
                    "avg_ms": int(avg_time),
                    "max_ms": int(max(times)),
                    "sufficient_data": False,
                },
            ))

    return findings


def _detect_error_spike(traces: list[dict]) -> list[Finding]:
    """Detect if tool errors are spiking relative to the baseline."""
    indicators = [1.0 if t["error_count"] > 0 else 0.0 for t in traces]
    total_errors = sum(t["error_count"] for t in traces)
    traces_with_errors = int(sum(indicators))

    if traces_with_errors == 0:
        return []

    findings = []
    bl = _compute_baseline(indicators)
    error_rate = _mean(indicators)

    if bl and bl["sufficient_data"]:
        if bl["baseline_stdev"] > 0:
            severity = _severity_from_z(bl["z_score"])
            if severity:
                findings.append(Finding(
                    pattern="error_spike",
                    severity=severity,
                    category="heal",
                    description=f"Error spike — {traces_with_errors}/{len(traces)} traces have errors ({error_rate:.0%}, {bl['z_score']:.1f}σ above baseline {bl['baseline_mean']:.0%}).",
                    evidence={
                        "traces_with_errors": traces_with_errors,
                        "total_errors": total_errors,
                        "error_rate": round(error_rate, 2),
                        "baseline_mean": round(bl["baseline_mean"], 2),
                        "baseline_stdev": round(bl["baseline_stdev"], 2),
                        "z_score": round(bl["z_score"], 2),
                        "sufficient_data": True,
                    },
                ))
        elif bl["baseline_mean"] == 0 and bl["recent_mean"] > 0:
            recent_errors = int(bl["recent_mean"] * 3)
            severity_map = {1: "low", 2: "medium"}
            severity = severity_map.get(recent_errors, "high")
            findings.append(Finding(
                pattern="error_spike",
                severity=severity,
                category="heal",
                description=f"New errors appearing — {recent_errors}/3 recent traces have errors vs zero in baseline.",
                evidence={
                    "traces_with_errors": traces_with_errors,
                    "total_errors": total_errors,
                    "error_rate": round(error_rate, 2),
                    "baseline_mean": 0.0,
                    "sufficient_data": True,
                },
            ))
    elif error_rate > 0.3:
        findings.append(Finding(
            pattern="error_spike",
            severity="high" if error_rate > 0.5 else "medium",
            category="heal",
            description=f"Tool errors in {traces_with_errors}/{len(traces)} traces ({error_rate:.0%}, limited data).",
            evidence={
                "traces_with_errors": traces_with_errors,
                "total_errors": total_errors,
                "error_rate": round(error_rate, 2),
                "sufficient_data": False,
            },
        ))

    return findings


def _detect_incomplete_pipeline(traces: list[dict]) -> list[Finding]:
    """Detect if the agent is skipping pipeline steps.

    Dynamically determines expected tools by finding tools that appear
    in >80% of traces — these are the agent's "standard" steps. Traces
    missing any standard tool are flagged as incomplete.
    """
    tool_sets = [t["unique_tools"] for t in traces if t["unique_tools"]]

    if len(tool_sets) < 3:
        return []

    tool_frequency: dict[str, int] = {}
    for ts in tool_sets:
        for tool in ts:
            tool_frequency[tool] = tool_frequency.get(tool, 0) + 1

    if len(tool_sets) < 5:
        freq_pct = 1.0
    elif len(tool_sets) <= 10:
        freq_pct = 0.8
    else:
        freq_pct = 0.7
    threshold = len(tool_sets) * freq_pct
    expected_tools = {tool for tool, count in tool_frequency.items() if count >= threshold}

    if not expected_tools:
        return []

    incomplete_count = 0
    missing_tools: dict[str, int] = {}

    for ts in tool_sets:
        missing = expected_tools - ts
        if missing:
            incomplete_count += 1
            for tool in missing:
                missing_tools[tool] = missing_tools.get(tool, 0) + 1

    if incomplete_count == 0:
        return []

    return [Finding(
        pattern="incomplete_pipeline",
        severity="medium",
        description=f"Pipeline incomplete in {incomplete_count}/{len(tool_sets)} traces. Missing steps: {', '.join(missing_tools.keys())}.",
        evidence={
            "incomplete_count": incomplete_count,
            "total_traces_with_tools": len(tool_sets),
            "expected_tools": sorted(expected_tools),
            "missing_tools": missing_tools,
            "frequency_threshold": freq_pct,
        },
    )]
