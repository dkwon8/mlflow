"""Periodic improve monitoring — scans experiments and runs analysis on a schedule."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

_logger = logging.getLogger(__name__)

_DEFAULT_INTERVAL_MINUTES = 1


def run_improve_monitoring_scheduler() -> None:
    """Scan experiments opted into auto-monitoring and run improve analysis.

    Called every minute by the Huey periodic task.  Each experiment has its
    own rate-limit check via the ``mlflow.improve.last_monitor_time`` tag so
    analysis only runs every N minutes (default 5).
    """
    from mlflow.client import MlflowClient
    from mlflow.server.handlers import _get_tracking_store

    tracking_store = _get_tracking_store()
    client = MlflowClient()

    experiments = client.search_experiments(
        filter_string="tags.`mlflow.improve.github_repo` != ''",
    )

    if not experiments:
        return

    _logger.debug("Improve monitor: found %d experiments with repo connected", len(experiments))

    for exp in experiments:
        try:
            _monitor_experiment(exp, client, tracking_store)
        except Exception:
            _logger.exception("Improve monitor: failed for experiment %s", exp.experiment_id)


def _monitor_experiment(exp, client, tracking_store) -> None:
    """Run a single monitoring cycle for one experiment."""
    from mlflow.entities.experiment_tag import ExperimentTag
    from mlflow.entities.issue import IssueSeverity
    from mlflow.genai.improve import analyze
    from mlflow.server.jobs import submit_job
    from mlflow.utils.mlflow_tags import (
        MLFLOW_IMPROVE_AUTO_FIX,
        MLFLOW_IMPROVE_GITHUB_REPO,
        MLFLOW_IMPROVE_LAST_MONITOR_TIME,
    )

    tags = exp.tags or {}
    repo_url = tags.get(MLFLOW_IMPROVE_GITHUB_REPO)
    if not repo_url:
        return

    interval_minutes = int(tags.get("mlflow.improve.monitor_interval_minutes", _DEFAULT_INTERVAL_MINUTES))
    last_time_str = tags.get(MLFLOW_IMPROVE_LAST_MONITOR_TIME)
    if last_time_str:
        last_time = datetime.fromisoformat(last_time_str)
        elapsed = (datetime.now(timezone.utc) - last_time).total_seconds() / 60
        if elapsed < interval_minutes:
            return

    _logger.info("Improve monitor: running analysis for experiment %s (%s)", exp.name, repo_url)

    now = datetime.now(timezone.utc).isoformat()
    tracking_store.set_experiment_tag(
        exp.experiment_id,
        ExperimentTag(MLFLOW_IMPROVE_LAST_MONITOR_TIME, now),
    )

    result = analyze(
        experiment_name=exp.name,
        mode="traces_only",
    )

    findings_count = result.get("summary", {}).get("findings_count", 0)
    if findings_count == 0:
        _logger.info("Improve monitor: no issues found for %s", exp.name)
        return

    last_patterns_raw = tags.get("mlflow.improve.last_patterns")
    last_patterns = json.loads(last_patterns_raw) if last_patterns_raw else []
    current_patterns = [f["pattern"] for f in result.get("findings", [])]
    new_patterns = [p for p in current_patterns if p not in last_patterns]

    _create_issues_for_suggestions(exp.experiment_id, result.get("suggestions", []))

    tracking_store.set_experiment_tag(
        exp.experiment_id,
        ExperimentTag("mlflow.improve.last_patterns", json.dumps(current_patterns)),
    )
    tracking_store.set_experiment_tag(
        exp.experiment_id,
        ExperimentTag("mlflow.improve.last_snapshot", json.dumps(result.get("summary", {}))),
    )

    if new_patterns:
        _logger.warning("Improve monitor: %d new issues in %s: %s", len(new_patterns), exp.name, new_patterns)

    _maybe_auto_fix(exp.experiment_id, result.get("suggestions", []), result.get("alerts", []))


_SEVERITY_MAP = {
    "high": "HIGH",
    "medium": "MEDIUM",
    "low": "LOW",
}


def _create_issues_for_suggestions(experiment_id: str, suggestions: list[dict]) -> None:
    """Create MLflow Issue entities for new suggestions."""
    from mlflow.entities.issue import IssueSeverity
    from mlflow.tracing.client import TracingClient

    tracing_client = TracingClient()

    for suggestion in suggestions:
        description = f"{suggestion['description']}\n\nRecommended action: {suggestion['action']}"
        try:
            tracing_client._create_issue(
                experiment_id=experiment_id,
                name=suggestion["title"],
                description=description,
                severity=getattr(IssueSeverity, _SEVERITY_MAP.get(suggestion["severity"], "MEDIUM")),
                categories=[f"[improve_{suggestion['type']}]", "[improve_monitor]"],
                root_causes=[
                    f"Confidence: {suggestion['confidence']:.0%}",
                    f"Pattern: {suggestion['id']}",
                ],
                created_by="mlflow.improve.monitor",
            )
        except Exception:
            _logger.debug("Failed to create issue for %s", suggestion.get("title"), exc_info=True)


def _maybe_auto_fix(experiment_id: str, suggestions: list[dict], alerts: list[dict] | None = None) -> None:
    """Submit fix jobs for heal-category suggestions automatically."""
    from mlflow.client import MlflowClient
    from mlflow.genai.improve.background_jobs import invoke_improve_fix_job
    from mlflow.server.jobs import submit_job

    alert_context = ""
    if alerts:
        alert_lines = []
        for a in alerts[:3]:
            alert_lines.append(f"- {a.get('failing_span', '?')}: {a.get('error_message', '')[:300]}")
        alert_context = "\n\nRecent errors from traces:\n" + "\n".join(alert_lines)

    client = MlflowClient()
    experiment = client.get_experiment(experiment_id)
    resolved_raw = (experiment.tags or {}).get("mlflow.improve.resolved_fixes", "[]")
    try:
        resolved_ids = {r.get("issue_id") for r in json.loads(resolved_raw)}
    except (ValueError, TypeError):
        resolved_ids = set()

    fixable = [
        s for s in suggestions
        if s.get("category") == "heal" and s.get("id") not in resolved_ids
    ]

    for s in fixable:
        _logger.info("Improve monitor: auto-healing '%s'", s["title"])
        description = s.get("description", "") + alert_context
        try:
            submit_job(
                invoke_improve_fix_job,
                {
                    "issue_id": s["id"],
                    "experiment_id": experiment_id,
                    "source": "auto",
                    "suggestion_title": s.get("title", ""),
                    "suggestion_description": description,
                    "suggestion_action": s.get("action", ""),
                },
            )
        except Exception:
            _logger.debug("Failed to submit auto-fix for %s", s.get("title"), exc_info=True)
