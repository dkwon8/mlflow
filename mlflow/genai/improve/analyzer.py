"""
Trace analyzer for the MLflow improve system.

Reads traces, tags, and assessments from MLflow and detects patterns
that indicate quality degradation, inefficiency, or scaling issues.

Each detection function returns a list of findings that the suggestion
engine uses to generate actionable fixes.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Finding:
    """A detected issue from trace analysis."""
    pattern: str
    severity: str  # "low", "medium", "high"
    description: str
    evidence: dict = field(default_factory=dict)


def analyze_traces(traces_data: list[dict]) -> list[Finding]:
    """Run all detection patterns against a set of traces.

    Args:
        traces_data: List of trace dicts, each with keys:
            - trace_id, tags (dict), assessments (list),
              execution_duration (int ms), trace_size (int bytes)

    Returns:
        List of Finding objects describing detected issues.
    """
    if not traces_data:
        return []

    findings = []
    findings.extend(_detect_context_bloat(traces_data))
    findings.extend(_detect_tool_redundancy(traces_data))
    findings.extend(_detect_score_degradation(traces_data))
    findings.extend(_detect_slowdown(traces_data))
    findings.extend(_detect_error_spike(traces_data))
    findings.extend(_detect_incomplete_pipeline(traces_data))
    return findings


def _detect_context_bloat(traces: list[dict]) -> list[Finding]:
    """Detect if trace sizes are growing, indicating context window pressure.

    Maps to mentor's example: 'if it has 1000 resumes, context bloat happens,
    switch from 250K model to 1M model.'
    """
    sizes = []
    for t in traces:
        size = int(t.get("tags", {}).get("agent.trace_size_bytes", 0))
        if size > 0:
            sizes.append(size)

    if len(sizes) < 2:
        return []

    findings = []
    avg_size = sum(sizes) / len(sizes)
    max_size = max(sizes)
    recent_avg = sum(sizes[:3]) / min(3, len(sizes))

    # Large traces (over 1MB suggest heavy context usage)
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

    # Growing trend (recent traces are bigger than older ones)
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
    """Detect if the agent is making redundant tool calls.

    Looks at agent.duplicate_tools tag across traces. If the same tools
    are duplicated repeatedly, it's a pattern worth fixing.
    """
    duplicate_counts: dict[str, int] = {}
    traces_with_duplicates = 0

    for t in traces:
        dupes = t.get("tags", {}).get("agent.duplicate_tools", "none")
        if dupes != "none" and dupes:
            traces_with_duplicates += 1
            for tool in dupes.split(", "):
                duplicate_counts[tool] = duplicate_counts.get(tool, 0) + 1

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
    """Detect if built-in assessment scores are trending downward.

    Tracks completeness, tool_call_correctness, tool_call_efficiency,
    and relevance_to_query over time.
    """
    score_map = {
        "completeness": [],
        "tool_call_correctness": [],
        "tool_call_efficiency": [],
        "relevance_to_query": [],
    }

    for t in traces:
        for a in t.get("assessments", []):
            name = a.get("name", "")
            if name in score_map:
                val = a.get("value")
                if val in ("yes", "true", "True", True):
                    score_map[name].append(1)
                elif val in ("no", "false", "False", False):
                    score_map[name].append(0)

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
    times = []
    for t in traces:
        ms = int(t.get("tags", {}).get("agent.execution_time_ms", 0))
        if ms > 0:
            times.append(ms)

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
    """Detect if tool errors are increasing."""
    error_counts = []
    for t in traces:
        errors = int(t.get("tags", {}).get("agent.tool_errors", 0))
        error_counts.append(errors)

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

    A full pipeline should use: parse_all_resumes, filter_candidates,
    score_all_candidates (or score_all_for_role), generate_report, sort_resumes.
    """
    expected_core = {"parse_all_resumes", "filter_candidates", "generate_report", "sort_resumes"}
    expected_scoring = {"score_all_candidates", "score_all_for_role"}

    incomplete_count = 0
    missing_tools: dict[str, int] = {}

    for t in traces:
        tools_used = t.get("tags", {}).get("agent.tools_used", "")
        if not tools_used or tools_used == "none":
            continue

        used_set = {tool.strip() for tool in tools_used.split(",")}

        # Only check traces that look like pipeline runs (have parse)
        if "parse_all_resumes" not in used_set:
            continue

        missing_core = expected_core - used_set
        has_scoring = bool(used_set & expected_scoring)

        if missing_core or not has_scoring:
            incomplete_count += 1
            for tool in missing_core:
                missing_tools[tool] = missing_tools.get(tool, 0) + 1
            if not has_scoring:
                missing_tools["scoring"] = missing_tools.get("scoring", 0) + 1

    if incomplete_count == 0:
        return []

    return [Finding(
        pattern="incomplete_pipeline",
        severity="medium",
        description=f"Pipeline incomplete in {incomplete_count} traces. Missing steps: {', '.join(missing_tools.keys())}.",
        evidence={
            "incomplete_count": incomplete_count,
            "missing_tools": missing_tools,
        },
    )]
