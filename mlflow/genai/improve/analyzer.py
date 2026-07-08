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


@dataclass
class Finding:
    """A detected issue from trace analysis."""
    pattern: str
    severity: str  # "low", "medium", "high"
    description: str
    evidence: dict = field(default_factory=dict)


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
    }


def analyze_traces(traces_data: list[dict]) -> list[Finding]:
    """Run all detection patterns against a set of traces.

    Args:
        traces_data: List of raw trace dicts from mlflow.search_traces().
            Each dict should have keys: spans, execution_duration, assessments.

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
    return findings


def _detect_context_bloat(traces: list[dict]) -> list[Finding]:
    """Detect if trace sizes are growing, indicating context window pressure."""
    sizes = [t["trace_size_bytes"] for t in traces if t["trace_size_bytes"] > 0]

    if len(sizes) < 2:
        return []

    findings = []
    avg_size = sum(sizes) / len(sizes)
    max_size = max(sizes)
    recent_avg = sum(sizes[:3]) / min(3, len(sizes))

    if max_size > 1_000_000:
        findings.append(Finding(
            pattern="context_bloat",
            severity="high" if max_size > 2_000_000 else "medium",
            description=f"Trace sizes averaging {avg_size / 1_000_000:.1f}MB, max {max_size / 1_000_000:.1f}MB. Large traces indicate heavy context window usage.",
            evidence={
                "avg_size_bytes": int(avg_size),
                "max_size_bytes": max_size,
                "recent_avg_bytes": int(recent_avg),
                "trace_count": len(sizes),
            },
        ))

    if len(sizes) >= 5:
        older_avg = sum(sizes[-3:]) / 3
        if recent_avg > older_avg * 1.5 and recent_avg > 500_000:
            findings.append(Finding(
                pattern="context_growth",
                severity="medium",
                description=f"Trace sizes growing — recent average {recent_avg / 1_000_000:.1f}MB vs older {older_avg / 1_000_000:.1f}MB.",
                evidence={
                    "recent_avg": int(recent_avg),
                    "older_avg": int(older_avg),
                    "growth_ratio": round(recent_avg / older_avg, 2),
                },
            ))

    return findings


def _detect_tool_redundancy(traces: list[dict]) -> list[Finding]:
    """Detect if the agent is making redundant tool calls by analyzing span data."""
    duplicate_counts: dict[str, int] = {}
    traces_with_duplicates = 0

    for t in traces:
        if t["duplicate_tools"]:
            traces_with_duplicates += 1
            for tool_name in t["duplicate_tools"]:
                duplicate_counts[tool_name] = duplicate_counts.get(tool_name, 0) + 1

    if not duplicate_counts:
        return []

    findings = []
    dupe_rate = traces_with_duplicates / len(traces)

    if dupe_rate > 0.3:
        worst_tool = max(duplicate_counts, key=duplicate_counts.get)
        findings.append(Finding(
            pattern="tool_redundancy",
            severity="medium" if dupe_rate > 0.5 else "low",
            description=f"Tool redundancy in {traces_with_duplicates}/{len(traces)} traces ({dupe_rate:.0%}). Most duplicated: {worst_tool}.",
            evidence={
                "duplicate_counts": duplicate_counts,
                "traces_affected": traces_with_duplicates,
                "rate": round(dupe_rate, 2),
            },
        ))

    return findings


def _detect_score_degradation(traces: list[dict]) -> list[Finding]:
    """Detect if assessment scores are trending downward."""
    score_map: dict[str, list[int]] = {}

    for t in traces:
        for a in t["assessments"]:
            name = a.get("name", "")
            if not name:
                continue
            val = a.get("value")
            if val in ("yes", "true", "True", True):
                score_map.setdefault(name, []).append(1)
            elif val in ("no", "false", "False", False):
                score_map.setdefault(name, []).append(0)

    findings = []
    for name, scores in score_map.items():
        if len(scores) < 3:
            continue

        pass_rate = sum(scores) / len(scores)
        recent_rate = sum(scores[:3]) / 3

        if pass_rate < 0.5:
            findings.append(Finding(
                pattern="score_degradation",
                severity="high" if pass_rate < 0.3 else "medium",
                description=f"{name} passing only {pass_rate:.0%} of the time ({sum(scores)}/{len(scores)} traces).",
                evidence={
                    "scorer": name,
                    "pass_rate": round(pass_rate, 2),
                    "recent_rate": round(recent_rate, 2),
                    "total_traces": len(scores),
                },
            ))

        if len(scores) >= 5:
            older_rate = sum(scores[-3:]) / 3
            if recent_rate < older_rate - 0.3:
                findings.append(Finding(
                    pattern="score_declining",
                    severity="medium",
                    description=f"{name} declining — recent {recent_rate:.0%} vs older {older_rate:.0%}.",
                    evidence={
                        "scorer": name,
                        "recent_rate": round(recent_rate, 2),
                        "older_rate": round(older_rate, 2),
                    },
                ))

    return findings


def _detect_slowdown(traces: list[dict]) -> list[Finding]:
    """Detect if execution time is increasing over time."""
    times = [t["execution_ms"] for t in traces if t["execution_ms"] > 0]

    if len(times) < 3:
        return []

    findings = []
    avg_time = sum(times) / len(times)
    recent_avg = sum(times[:3]) / 3

    if avg_time > 120_000:
        findings.append(Finding(
            pattern="slow_execution",
            severity="medium" if avg_time > 180_000 else "low",
            description=f"Average execution time is {avg_time / 1000:.0f}s. Pipeline runs over 2 minutes.",
            evidence={
                "avg_ms": int(avg_time),
                "recent_avg_ms": int(recent_avg),
                "max_ms": max(times),
            },
        ))

    if len(times) >= 5:
        older_avg = sum(times[-3:]) / 3
        if recent_avg > older_avg * 1.5:
            findings.append(Finding(
                pattern="execution_slowdown",
                severity="medium",
                description=f"Execution slowing — recent {recent_avg / 1000:.0f}s vs older {older_avg / 1000:.0f}s.",
                evidence={
                    "recent_avg_ms": int(recent_avg),
                    "older_avg_ms": int(older_avg),
                    "ratio": round(recent_avg / older_avg, 2),
                },
            ))

    return findings


def _detect_error_spike(traces: list[dict]) -> list[Finding]:
    """Detect if tool errors are increasing by counting error spans."""
    error_counts = [t["error_count"] for t in traces]

    if not error_counts:
        return []

    total_errors = sum(error_counts)
    traces_with_errors = sum(1 for e in error_counts if e > 0)

    if traces_with_errors == 0:
        return []

    findings = []
    error_rate = traces_with_errors / len(error_counts)

    if error_rate > 0.2:
        findings.append(Finding(
            pattern="error_spike",
            severity="high" if error_rate > 0.5 else "medium",
            description=f"Tool errors in {traces_with_errors}/{len(error_counts)} traces ({error_rate:.0%}). Total errors: {total_errors}.",
            evidence={
                "traces_with_errors": traces_with_errors,
                "total_errors": total_errors,
                "error_rate": round(error_rate, 2),
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

    threshold = len(tool_sets) * 0.8
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
        },
    )]
