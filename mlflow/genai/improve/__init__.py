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


__all__ = ["analyze", "analyze_traces", "generate_suggestions", "Finding", "Suggestion"]
