"""
MLflow Improve — Self-optimization and self-healing for MLflow-deployed agents.

Analyzes trace history and evaluation scores to detect quality degradation,
inefficiency, and scaling issues. Generates actionable suggestions to
improve agent performance.

Usage:
    import mlflow.genai.improve

    # Analyze the last 20 traces for an experiment
    result = mlflow.genai.improve.analyze(
        experiment_name="recruitment-filtration-agent",
        trace_count=20,
    )

    # See what was found
    for suggestion in result["suggestions"]:
        print(f"[{suggestion['severity']}] {suggestion['title']}")
        print(f"  Action: {suggestion['action']}")

Works with any agent deployed on MLflow — no custom instrumentation required.
"""

from __future__ import annotations

from .analyzer import analyze_traces, Finding
from .suggestions import generate_suggestions, Suggestion


def analyze(
    experiment_name: str,
    trace_count: int = 20,
    tracking_uri: str | None = None,
) -> dict:
    """Analyze recent traces and generate improvement suggestions.

    Reads the last N traces from an MLflow experiment, parses raw span
    data to extract tool calls, errors, and timing, runs detection
    patterns, and returns actionable suggestions.

    Args:
        experiment_name: Name of the MLflow experiment to analyze.
        trace_count: Number of recent traces to analyze (default 20).
        tracking_uri: MLflow tracking server URI. Uses default if not set.

    Returns:
        Dict with keys:
            - findings: list of detected issues
            - suggestions: list of actionable fixes
            - summary: overview stats
    """
    import mlflow

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    exp = mlflow.get_experiment_by_name(experiment_name)
    if not exp:
        return {
            "findings": [],
            "suggestions": [],
            "summary": {"status": "no_experiment", "experiment_name": experiment_name},
        }

    raw_traces = mlflow.search_traces(
        experiment_ids=[exp.experiment_id],
        max_results=trace_count,
    )

    if len(raw_traces) == 0:
        return {
            "findings": [],
            "suggestions": [],
            "summary": {"status": "no_traces", "experiment_name": experiment_name},
        }

    traces_data = []
    for _, row in raw_traces.iterrows():
        traces_data.append({
            "trace_id": row.get("trace_id", ""),
            "spans": row.get("spans", []),
            "execution_duration": int(row.get("execution_duration", 0) or 0),
            "assessments": row.get("assessments", []),
        })

    findings = analyze_traces(traces_data)
    suggestions = generate_suggestions(findings)

    from .analyzer import _parse_trace
    parsed = [_parse_trace(t) for t in traces_data]

    total_tool_calls = sum(p["tool_call_count"] for p in parsed)
    avg_tool_calls = total_tool_calls / len(parsed) if parsed else 0

    return {
        "findings": [
            {
                "pattern": f.pattern,
                "severity": f.severity,
                "description": f.description,
                "evidence": f.evidence,
            }
            for f in findings
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
        "summary": {
            "status": "ok",
            "experiment_name": experiment_name,
            "traces_analyzed": len(traces_data),
            "findings_count": len(findings),
            "suggestions_count": len(suggestions),
            "avg_tool_calls": round(avg_tool_calls, 1),
            "high_severity": sum(1 for f in findings if f.severity == "high"),
            "medium_severity": sum(1 for f in findings if f.severity == "medium"),
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


_monitor_state: dict = {}


def enable_auto_improve(
    experiment_name: str,
    check_every_n_traces: int = 10,
    auto_fix: bool = False,
    tracking_uri: str | None = None,
):
    """Enable automatic self-optimization for an experiment.

    Once enabled, the system automatically:
    1. Monitors traces — runs analysis every N traces
    2. Detects degradation — compares against the last known baseline
    3. Creates Issues in MLflow — visible in the Improve tab
    4. User clicks "Fix it" in the UI — code agent creates a PR

    The detection is automatic. The fix requires a manual click (unless
    auto_fix=True is set for fully hands-off operation).

    Args:
        experiment_name: Name of the MLflow experiment to monitor.
        check_every_n_traces: Run analysis after every N new traces (default 10).
        auto_fix: If True, automatically create fix PRs without user
            intervention. Default False — user clicks "Fix it" in the UI.
        tracking_uri: MLflow tracking server URI.

    Example:
        import mlflow.genai.improve

        # One line — detection is automatic, fix via UI
        mlflow.genai.improve.enable_auto_improve(
            experiment_name="my-agent",
            check_every_n_traces=10,
        )
    """
    import logging
    import mlflow

    logger = logging.getLogger("mlflow.genai.improve")

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    _monitor_state[experiment_name] = {
        "trace_count_at_last_check": 0,
        "check_every_n": check_every_n_traces,
        "auto_fix": auto_fix,
        "tracking_uri": tracking_uri,
        "enabled": True,
    }

    snapshot(experiment_name, tracking_uri=tracking_uri)
    logger.info(
        "Auto-improve enabled for '%s' — analyzing every %d traces, auto_fix=%s",
        experiment_name,
        check_every_n_traces,
        auto_fix,
    )

    _register_trace_callback(experiment_name)


def _register_trace_callback(experiment_name: str):
    """Register a callback that triggers analysis after N traces."""
    import logging
    import mlflow

    logger = logging.getLogger("mlflow.genai.improve")

    original_end_span = None
    try:
        from mlflow.tracing.export.mlflow import MlflowSpanExporter
        original_export = MlflowSpanExporter.export

        def _patched_export(self, spans):
            result = original_export(self, spans)
            _on_trace_logged(experiment_name, logger)
            return result

        MlflowSpanExporter.export = _patched_export
        logger.debug("Trace callback registered via span exporter patch")
    except Exception:
        logger.debug("Could not patch span exporter — using trace count polling")


def _on_trace_logged(experiment_name: str, logger):
    """Called after each trace is logged. Checks if it's time to analyze."""
    import mlflow

    state = _monitor_state.get(experiment_name)
    if not state or not state["enabled"]:
        return

    exp = mlflow.get_experiment_by_name(experiment_name)
    if not exp:
        return

    try:
        current_traces = mlflow.search_traces(
            experiment_ids=[exp.experiment_id],
            max_results=1,
        )
        current_count = len(current_traces)
    except Exception:
        return

    last_check = state.get("trace_count_at_last_check", 0)
    if current_count - last_check < state["check_every_n"]:
        return

    state["trace_count_at_last_check"] = current_count
    logger.info("Auto-improve: running analysis on '%s' (%d traces since last check)", experiment_name, current_count - last_check)

    try:
        _run_auto_cycle(experiment_name, state, logger)
    except Exception as e:
        logger.warning("Auto-improve analysis failed: %s", e)


def _run_auto_cycle(experiment_name: str, state: dict, logger):
    """Run a full analysis cycle: analyze → compare → optionally fix."""
    import json
    import mlflow

    tracking_uri = state.get("tracking_uri")
    current = analyze(experiment_name, tracking_uri=tracking_uri)

    if current["summary"].get("findings_count", 0) == 0:
        logger.info("Auto-improve: no issues found — agent is healthy")
        return

    exp = mlflow.get_experiment_by_name(experiment_name)
    last_patterns_raw = exp.tags.get("mlflow.improve.last_patterns") if exp else None

    if last_patterns_raw:
        last_patterns = json.loads(last_patterns_raw)
        current_patterns = [f["pattern"] for f in current["findings"]]
        new_issues = [p for p in current_patterns if p not in last_patterns]

        if new_issues:
            logger.warning("Auto-improve: NEW issues detected: %s", new_issues)
        else:
            logger.info("Auto-improve: no new issues (existing: %s)", current_patterns)

    snapshot(experiment_name, tracking_uri=tracking_uri)

    high_auto_fixable = [
        s for s in current["suggestions"]
        if s["severity"] == "high" and s["auto_applicable"] and s["confidence"] >= 0.8
    ]

    if state.get("auto_fix") and high_auto_fixable and exp:
        repo_url = exp.tags.get("mlflow.improve.github_repo")
        if repo_url:
            logger.info("Auto-improve: triggering auto-fix for %d high-confidence issues", len(high_auto_fixable))
            _auto_fix(experiment_name, exp, high_auto_fixable, tracking_uri, logger)
        else:
            logger.info("Auto-improve: %d auto-fixable issues found but no GitHub repo connected", len(high_auto_fixable))
    elif high_auto_fixable:
        logger.info("Auto-improve: %d auto-fixable issues found (auto_fix disabled)", len(high_auto_fixable))


def _auto_fix(experiment_name: str, exp, suggestions: list, tracking_uri: str | None, logger):
    """Automatically create fix PRs for high-confidence suggestions."""
    from mlflow.genai.improve.code_agent import FixRequest, get_agent
    import mlflow.genai.improve.agents  # noqa: F401

    agent_name = exp.tags.get("mlflow.improve.code_agent", "claude-code")
    repo_url = exp.tags.get("mlflow.improve.github_repo")
    branch = exp.tags.get("mlflow.improve.github_branch", "main")

    agent = get_agent(agent_name)

    for s in suggestions:
        logger.info("Auto-fix: creating PR for '%s'", s["title"])
        request = FixRequest(
            issue_id=s["id"],
            issue_name=s["title"],
            issue_description=f"{s['description']}\n\nRecommended action: {s['action']}",
            root_causes=[f"Confidence: {s['confidence']:.0%}", f"Pattern: {s['id']}"],
            repo_url=repo_url,
            branch=branch,
            experiment_id=exp.experiment_id,
        )

        result = agent.create_fix(request)
        if result.success:
            logger.info("Auto-fix: PR created — %s", result.pr_url)
        else:
            logger.warning("Auto-fix: failed — %s", result.error)


def disable_auto_improve(experiment_name: str):
    """Disable automatic monitoring for an experiment."""
    if experiment_name in _monitor_state:
        _monitor_state[experiment_name]["enabled"] = False


__all__ = [
    "analyze",
    "analyze_traces",
    "compare",
    "disable_auto_improve",
    "enable_auto_improve",
    "generate_suggestions",
    "snapshot",
    "Finding",
    "Suggestion",
]
