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
        print(f"[{suggestion.severity}] {suggestion.title}")
        print(f"  Action: {suggestion.action}")

Works with any agent deployed on MLflow — not specific to any one agent.
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

    Reads the last N traces from an MLflow experiment, runs detection
    patterns (context bloat, tool redundancy, score degradation, etc.),
    and returns actionable suggestions.

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
        trace_id = row.get("trace_id", "")

        tags = row.get("tags", {})
        if not isinstance(tags, dict):
            tags = {}

        assessments = []
        raw_assessments = row.get("assessments", [])
        if isinstance(raw_assessments, list):
            for a in raw_assessments:
                if isinstance(a, dict):
                    assessments.append(a)
                else:
                    assessments.append({
                        "name": getattr(a, "assessment_name", getattr(a, "name", "")),
                        "value": getattr(a, "string_value", getattr(a, "value", "")),
                    })

        traces_data.append({
            "trace_id": trace_id,
            "tags": tags,
            "assessments": assessments,
            "execution_duration": int(row.get("execution_duration", 0) or 0),
        })

    findings = analyze_traces(traces_data)
    suggestions = generate_suggestions(findings)

    model_tags = [t["tags"].get("agent.model", "") for t in traces_data if t["tags"].get("agent.model")]
    current_model = model_tags[0] if model_tags else "unknown"

    avg_tool_calls = 0
    tool_counts = [int(t["tags"].get("agent.tool_call_count", 0)) for t in traces_data if t["tags"].get("agent.tool_call_count")]
    if tool_counts:
        avg_tool_calls = sum(tool_counts) / len(tool_counts)

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
            "current_model": current_model,
            "avg_tool_calls": round(avg_tool_calls, 1),
            "high_severity": sum(1 for f in findings if f.severity == "high"),
            "medium_severity": sum(1 for f in findings if f.severity == "medium"),
        },
    }


__all__ = ["analyze", "analyze_traces", "generate_suggestions", "Finding", "Suggestion"]
