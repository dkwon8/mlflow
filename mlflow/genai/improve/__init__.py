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

MIN_TRACES = 10

_ERROR_TERM_PATTERNS = [
    r"model[_ ]not[_ ]found[:\s]+['\"]?([a-zA-Z0-9._-]+)",
    r"(?:unknown|invalid|unsupported)\s+model[:\s]+['\"]?([a-zA-Z0-9._-]+)",
    r"(?:requested\s+)?model\s+['\"]([a-zA-Z0-9._-]+)['\"]",
    r"(?:KeyError|NameError|AttributeError)[:\s]+['\"]?([a-zA-Z0-9_.]+)",
    r"No module named ['\"]([a-zA-Z0-9_.]+)",
    r"does not exist.*['\"]([a-zA-Z0-9._-]{5,})['\"]",
    r"['\"]([a-zA-Z0-9._-]{5,})['\"].*does not exist",
]


def _check_errors_against_code(
    error_sigs: set[str], repo_dir: "Path"
) -> set[str]:
    """Check which error signatures have been fixed in the current code.

    Extracts searchable terms from error messages (model names, key names,
    module names) and greps the repo. If the term no longer appears in the
    code, the error is likely resolved.
    """
    import re
    import subprocess

    resolved = set()
    for sig in error_sigs:
        error_msg = sig.split(":", 1)[1] if ":" in sig else sig

        _noise = {"error", "type", "message", "none", "null", "true", "false", "code", "status", "data", "value", "result"}
        search_terms: list[str] = []
        for pattern in _ERROR_TERM_PATTERNS:
            matches = re.findall(pattern, error_msg, re.IGNORECASE)
            search_terms.extend(m for m in matches if m.lower() not in _noise)

        if not search_terms:
            continue

        all_absent = True
        for term in search_terms:
            if len(term) < 4:
                continue
            try:
                result = subprocess.run(
                    ["grep", "-rl", "--include=*.py", "--include=*.yaml",
                     "--include=*.yml", "--include=*.json", "--include=*.toml",
                     "--include=*.env", term, str(repo_dir)],
                    capture_output=True, text=True, timeout=10,
                )
                if result.returncode == 0 and result.stdout.strip():
                    all_absent = False
                    break
            except (subprocess.TimeoutExpired, FileNotFoundError):
                all_absent = False
                break

        if all_absent and search_terms:
            resolved.add(sig)
            _logger.info("Error signature resolved in code: %s (term '%s' absent)", sig, search_terms[0])

    return resolved


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

        if len(raw_traces) > 0 and len(raw_traces) < MIN_TRACES:
            return {
                "findings": [],
                "code_findings": [],
                "suggestions": [],
                "alerts": [],
                "summary": {
                    "status": "insufficient_traces",
                    "experiment_name": experiment_name,
                    "traces_available": len(raw_traces),
                    "traces_required": MIN_TRACES,
                },
            }

        if len(raw_traces) > 0:
            for _, row in raw_traces.iterrows():
                traces_data.append({
                    "trace_id": row.get("trace_id", ""),
                    "spans": row.get("spans", []),
                    "execution_duration": int(row.get("execution_duration", 0) or 0),
                    "assessments": row.get("assessments", []),
                })

            analysis_mode = mode if mode in ("heal", "improve") else None
            trace_finding_objs = analyze_traces(traces_data, mode=analysis_mode)
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

    _CODE_HEAL_PATTERNS = {"missing_error_handling", "anti_pattern", "security"}

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

    parsed = [_parse_trace(t) for t in traces_data] if traces_data else []

    total_tool_calls = sum(p["tool_call_count"] for p in parsed)
    avg_tool_calls = total_tool_calls / len(parsed) if parsed else 0
    error_count = sum(1 for p in parsed if p["error_count"] > 0)
    healthy_count = len(parsed) - error_count
    latencies = [p["execution_ms"] for p in parsed if p["execution_ms"] > 0]
    avg_latency_ms = round(sum(latencies) / len(latencies)) if latencies else 0

    recency_window = min(5, len(parsed))
    recent_error_sigs: set[str] = set()
    for p in parsed[:recency_window]:
        for err in p.get("error_details", []):
            sig = f"{err['span_name']}:{(err['error_message'] or '')[:200]}"
            recent_error_sigs.add(sig)

    repo_dir = None
    if has_repo and repo_url:
        try:
            repo_dir = clone_or_fetch_repo(repo_url, branch)
        except Exception:
            pass

    resolved_in_code: set[str] = set()
    if repo_dir:
        resolved_in_code = _check_errors_against_code(recent_error_sigs, repo_dir)

    raw_alerts: list[dict] = []
    for p in parsed:
        if p["error_details"]:
            first_error = p["error_details"][0]
            sig = f"{first_error['span_name']}:{(first_error['error_message'] or '')[:200]}"
            if sig not in recent_error_sigs:
                continue
            if sig in resolved_in_code:
                continue

            timestamp = None
            if p.get("start_ns") and p["start_ns"] > 0:
                from datetime import datetime, timezone
                timestamp = datetime.fromtimestamp(
                    p["start_ns"] / 1e9, tz=timezone.utc
                ).isoformat()

            raw_alerts.append({
                "trace_id": p["trace_id"],
                "error_message": first_error["error_message"] or "Unknown error",
                "user_query": p["user_query"] or "",
                "failing_span": first_error["span_name"],
                "severity": "high",
                "timestamp": timestamp,
                "_sig": sig,
            })

    seen_sigs: set[str] = set()
    alerts: list[dict] = []
    for a in raw_alerts:
        if a["_sig"] not in seen_sigs:
            seen_sigs.add(a["_sig"])
            alert = {k: v for k, v in a.items() if k != "_sig"}
            alerts.append(alert)

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
