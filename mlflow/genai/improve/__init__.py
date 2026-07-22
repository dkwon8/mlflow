"""
MLflow Improve — continuous diagnostics for MLflow-deployed agents.

Analyzes an agent's traces and codebase together to detect problems
and suggest fixes. Works with any agent framework that logs traces
to MLflow.

Usage:
    from mlflow.genai.improve import analyze

    result = analyze(experiment_name="my-agent", repo_url="owner/repo")

    for s in result["suggestions"]:
        print(f"[{s['severity']}] {s['title']}: {s['action']}")
"""

from __future__ import annotations

import logging

from .trace_analyzer import analyze_traces, Finding, _parse_trace
from .code_analyzer import CodeFinding, analyze_code
from .github_fetcher import fetch_repo_files
from .suggestions import generate_suggestions, Suggestion
from .summary import compute_alerts, compute_summary

_logger = logging.getLogger(__name__)

MIN_TRACES = 10

_CODE_HEAL_PATTERNS = {"missing_error_handling", "anti_pattern", "security"}


def analyze(
    experiment_name: str,
    trace_count: int = 20,
    tracking_uri: str | None = None,
    repo_url: str | None = None,
    branch: str = "main",
    model: str = "openai:/gpt-5.4-mini",
) -> dict:
    """Analyze an agent's traces and codebase to diagnose problems and suggest fixes.

    Always analyzes both traces and code together. Requires:
    - A GitHub repository connected to the experiment
    - At least 10 traces logged

    Args:
        experiment_name: Name of the MLflow experiment to analyze.
        trace_count: Number of recent traces to analyze (default 20).
        tracking_uri: MLflow tracking server URI. Uses default if not set.
        repo_url: GitHub repo URL or owner/repo shorthand. Falls back to
            the experiment's mlflow.improve.github_repo tag if not provided.
        branch: Branch to analyze (default "main").
        model: Model URI for LLM code analysis (default "openai:/gpt-5.4-mini").

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
        return _empty_result("no_experiment", experiment_name)

    if not repo_url:
        repo_url = exp.tags.get("mlflow.improve.github_repo")

    if not repo_url:
        return _empty_result("no_repo", experiment_name,
                             error="Connect a GitHub repository to use the improve feature.")

    sibling_exps = mlflow.search_experiments(
        filter_string=f"tags.`mlflow.improve.github_repo` = '{repo_url}'"
    )
    all_exp_ids = [e.experiment_id for e in sibling_exps] if sibling_exps else [exp.experiment_id]
    if exp.experiment_id not in all_exp_ids:
        all_exp_ids.append(exp.experiment_id)

    raw_traces = mlflow.search_traces(
        experiment_ids=all_exp_ids,
        max_results=trace_count,
    )

    if len(raw_traces) == 0:
        return _empty_result("no_traces", experiment_name,
                             error="No traces found. Run your agent to generate traces first.")

    if len(raw_traces) < MIN_TRACES:
        return _empty_result("insufficient_traces", experiment_name,
                             traces_available=len(raw_traces),
                             traces_required=MIN_TRACES,
                             error=f"Need at least {MIN_TRACES} traces (currently {len(raw_traces)}).")

    traces_data = []
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
            "category": f.category,
            "description": f.description,
            "evidence": f.evidence,
        }
        for f in trace_finding_objs
    ]

    code_findings_list: list[CodeFinding] = []
    selected_files: list[tuple[str, str]] = []
    try:
        _logger.info("Fetching code from %s via GitHub API", repo_url)

        hint_parsed = [_parse_trace(t) for t in traces_data]
        trace_hints = list({
            tool for p in hint_parsed for tool in p["tool_names"]
        })

        selected_files = fetch_repo_files(repo_url, branch, trace_hints=trace_hints)
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
            category="heal" if cf.pattern in _CODE_HEAL_PATTERNS else "improve",
            description=cf.description,
            evidence={
                **(cf.evidence or {}),
                "file_path": cf.file_path,
                "root_cause": cf.root_cause,
                "suggested_fix": cf.suggested_fix,
            },
        ))

    suggestions = generate_suggestions(all_finding_objs)

    parsed = [_parse_trace(t) for t in traces_data]

    alerts = compute_alerts(parsed, file_contents=selected_files if selected_files else None)
    summary = compute_summary(experiment_name, parsed, trace_findings, code_findings_list, repo_url)
    summary["experiments_pooled"] = len(all_exp_ids)

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
                "category": s.category,
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
        "summary": summary,
    }


def _empty_result(status: str, experiment_name: str, **extra) -> dict:
    """Return an empty analysis result with the given status."""
    return {
        "findings": [],
        "code_findings": [],
        "suggestions": [],
        "alerts": [],
        "summary": {"status": status, "experiment_name": experiment_name, **extra},
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
