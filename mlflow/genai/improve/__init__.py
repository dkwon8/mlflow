"""
MLflow Improve — self-optimization for MLflow-deployed agents.

Three analysis layers:
  1. Statistical baseline detectors (z-score anomaly detection on traces)
  2. LLM-as-judge scorers (MLflow built-in: completeness, correctness, etc.)
  3. LLM code analyzer (clones repo, reads source, finds issues)

Usage:
    from mlflow.genai.improve import analyze

    result = analyze(experiment_name="my-agent", repo_url="owner/repo")

    for s in result["suggestions"]:
        print(f"[{s['severity']}] {s['title']}: {s['action']}")
"""

from __future__ import annotations

import logging

from .trace_analyzer import analyze_traces, Finding, _parse_trace
from .code_analyzer import (
    CodeFinding,
    analyze_code,
    clone_or_fetch_repo,
    select_relevant_files,
)
from .suggestions import generate_suggestions, Suggestion

_logger = logging.getLogger(__name__)


def analyze(
    experiment_name: str,
    trace_count: int = 20,
    tracking_uri: str | None = None,
    repo_url: str | None = None,
    branch: str = "main",
    model: str = "openai:/gpt-5.4-mini",
    mode: str = "auto",
) -> dict:
    """Analyze traces and/or repository code to generate improvement suggestions.

    Supports three analysis modes:
    - "traces_only": Traditional rule-based trace analysis (existing behavior)
    - "code_only": LLM-powered analysis of the connected repo (no traces needed)
    - "both": Run both trace and code analysis, merge findings
    - "auto": If traces exist, run both; if no traces, run code-only

    Args:
        experiment_name: Name of the MLflow experiment to analyze.
        trace_count: Number of recent traces to analyze (default 20).
        tracking_uri: MLflow tracking server URI. Uses default if not set.
        repo_url: GitHub repo URL or owner/repo shorthand. Falls back to
            the experiment's mlflow.improve.github_repo tag if not provided.
        branch: Branch to analyze (default "main").
        model: Model URI for LLM code analysis (default "openai:/gpt-5.4-mini").
        mode: Analysis mode — "auto", "traces_only", "code_only", or "both".

    Returns:
        Dict with keys:
            - findings: list of trace-based detected issues
            - code_findings: list of code-based detected issues
            - suggestions: list of actionable fixes (from both sources)
            - alerts: list of error alerts from traces
            - summary: overview stats
    """
    import mlflow

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    exp = mlflow.get_experiment_by_name(experiment_name)
    if not exp:
        return {
            "findings": [],
            "code_findings": [],
            "suggestions": [],
            "alerts": [],
            "summary": {"status": "no_experiment", "experiment_name": experiment_name},
        }

    if not repo_url:
        repo_url = exp.tags.get("mlflow.improve.github_repo")

    has_repo = bool(repo_url)

    trace_finding_objs: list[Finding] = []
    trace_findings: list[dict] = []
    traces_data: list[dict] = []

    run_traces = mode in ("traces_only", "both") or (mode == "auto")
    run_code = mode in ("code_only", "both") or (mode == "auto" and has_repo)

    if run_traces:
        raw_traces = mlflow.search_traces(
            experiment_ids=[exp.experiment_id],
            max_results=trace_count,
        )

        if len(raw_traces) > 0:
            for _, row in raw_traces.iterrows():
                traces_data.append({
                    "trace_id": row.get("trace_id", ""),
                    "spans": row.get("spans", []),
                    "execution_duration": int(row.get("execution_duration", 0) or 0),
                    "assessments": row.get("assessments", []),
                })

            trace_finding_objs = analyze_traces(traces_data)
            trace_findings = [
                {
                    "pattern": f.pattern,
                    "severity": f.severity,
                    "description": f.description,
                    "evidence": f.evidence,
                }
                for f in trace_finding_objs
            ]
        elif mode == "auto" and has_repo:
            run_code = True

    code_findings_list: list[CodeFinding] = []
    if run_code and has_repo:
        try:
            _logger.info("Running dynamic code analysis on %s", repo_url)
            repo_dir = clone_or_fetch_repo(repo_url, branch)

            trace_hints = None
            if traces_data:
                hint_parsed = [_parse_trace(t) for t in traces_data]
                trace_hints = list({
                    tool for p in hint_parsed for tool in p["tool_names"]
                })

            selected_files = select_relevant_files(repo_dir, trace_hints=trace_hints)
            code_findings_list = analyze_code(
                selected_files,
                trace_findings=trace_findings if trace_findings else None,
                model=model,
            )
            _logger.info("Code analysis found %d issues", len(code_findings_list))
        except Exception:
            _logger.exception("Dynamic code analysis failed")

    all_finding_objs = list(trace_finding_objs)
    for cf in code_findings_list:
        all_finding_objs.append(Finding(
            pattern=cf.pattern,
            severity=cf.severity,
            description=cf.description,
            evidence={
                **(cf.evidence or {}),
                "file_path": cf.file_path,
                "root_cause": cf.root_cause,
                "suggested_fix": cf.suggested_fix,
            },
        ))

    suggestions = generate_suggestions(all_finding_objs)

    parsed = [_parse_trace(t) for t in traces_data] if traces_data else []

    total_tool_calls = sum(p["tool_call_count"] for p in parsed)
    avg_tool_calls = total_tool_calls / len(parsed) if parsed else 0
    error_count = sum(1 for p in parsed if p["error_count"] > 0)
    healthy_count = len(parsed) - error_count
    latencies = [p["execution_ms"] for p in parsed if p["execution_ms"] > 0]
    avg_latency_ms = round(sum(latencies) / len(latencies)) if latencies else 0

    alerts = []
    for p in parsed:
        if p["error_details"]:
            first_error = p["error_details"][0]
            alerts.append({
                "trace_id": p["trace_id"],
                "error_message": first_error["error_message"] or "Unknown error",
                "user_query": p["user_query"] or "",
                "failing_span": first_error["span_name"],
                "severity": "high",
            })

    return {
        "findings": trace_findings,
        "code_findings": [
            {
                "pattern": cf.pattern,
                "severity": cf.severity,
                "description": cf.description,
                "file_path": cf.file_path,
                "root_cause": cf.root_cause,
                "suggested_fix": cf.suggested_fix,
                "confidence": cf.confidence,
                "evidence": cf.evidence,
            }
            for cf in code_findings_list
        ],
        "suggestions": [
            {
                "id": s.id,
                "type": s.type,
                "severity": s.severity,
                "title": s.title,
                "description": s.description,
                "action": s.action,
                "confidence": s.confidence,
                "auto_applicable": s.auto_applicable,
                "evidence": s.evidence,
            }
            for s in suggestions
        ],
        "alerts": alerts,
        "summary": {
            "status": "ok",
            "experiment_name": experiment_name,
            "traces_analyzed": len(traces_data),
            "total_traces": len(parsed) if parsed else 0,
            "healthy_count": healthy_count,
            "error_count": error_count,
            "avg_latency_ms": avg_latency_ms,
            "findings_count": len(trace_findings) + len(code_findings_list),
            "code_findings_count": len(code_findings_list),
            "suggestions_count": len(suggestions),
            "avg_tool_calls": round(avg_tool_calls, 1),
            "high_severity": sum(
                1 for f in trace_findings if f.get("severity") == "high"
            ) + sum(
                1 for f in code_findings_list if f.severity == "high"
            ),
            "medium_severity": sum(
                1 for f in trace_findings if f.get("severity") == "medium"
            ) + sum(
                1 for f in code_findings_list if f.severity == "medium"
            ),
            "analysis_mode": mode,
            "repo_analyzed": has_repo and run_code,
        },
    }


def compare(before: dict, after: dict) -> dict:
    """Compare two analysis results to see what improved or regressed.

    Args:
        before: Result from analyze() before a fix was applied.
        after: Result from analyze() after the fix.

    Returns:
        Dict with resolved, new, and persistent findings, plus metric deltas.
    """
    before_patterns = {f["pattern"]: f for f in before.get("findings", [])}
    after_patterns = {f["pattern"]: f for f in after.get("findings", [])}

    resolved = []
    for pattern, finding in before_patterns.items():
        if pattern not in after_patterns:
            resolved.append({"pattern": pattern, "was": finding})

    new_issues = []
    for pattern, finding in after_patterns.items():
        if pattern not in before_patterns:
            new_issues.append({"pattern": pattern, "finding": finding})

    persistent = []
    for pattern in before_patterns:
        if pattern in after_patterns:
            persistent.append({
                "pattern": pattern,
                "before": before_patterns[pattern],
                "after": after_patterns[pattern],
            })

    before_summary = before.get("summary", {})
    after_summary = after.get("summary", {})

    return {
        "resolved": resolved,
        "new_issues": new_issues,
        "persistent": persistent,
        "metrics": {
            "findings_before": before_summary.get("findings_count", 0),
            "findings_after": after_summary.get("findings_count", 0),
            "high_severity_before": before_summary.get("high_severity", 0),
            "high_severity_after": after_summary.get("high_severity", 0),
            "avg_tool_calls_before": before_summary.get("avg_tool_calls", 0),
            "avg_tool_calls_after": after_summary.get("avg_tool_calls", 0),
        },
        "improved": len(resolved) > 0 and len(new_issues) == 0,
    }


def snapshot(
    experiment_name: str,
    trace_count: int = 20,
    tracking_uri: str | None = None,
) -> dict:
    """Run analysis and save the result as an experiment tag for later comparison.

    This creates a baseline that compare() can use after fixes are applied.

    Args:
        experiment_name: Name of the MLflow experiment.
        trace_count: Number of traces to analyze.
        tracking_uri: MLflow tracking server URI.

    Returns:
        The analysis result (also saved as experiment tag).
    """
    import json
    import mlflow

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    result = analyze(experiment_name, trace_count, tracking_uri)

    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp:
        client = mlflow.MlflowClient()
        client.set_experiment_tag(
            exp.experiment_id,
            "mlflow.improve.last_snapshot",
            json.dumps(result["summary"]),
        )
        patterns = [f["pattern"] for f in result["findings"]]
        client.set_experiment_tag(
            exp.experiment_id,
            "mlflow.improve.last_patterns",
            json.dumps(patterns),
        )

    return result


__all__ = [
    "analyze",
    "compare",
    "snapshot",
]
