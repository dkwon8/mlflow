"""Periodic improve monitoring — scans experiments and runs analysis on a schedule.

Every 10 minutes, the Huey periodic task calls run_improve_monitoring_scheduler().
It finds all experiments with a connected GitHub repo, runs the full analysis
(traces + codebase), and creates MLflow Issue entities as notifications.
Engineers see these in the Improve tab and decide whether to act.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

_logger = logging.getLogger(__name__)

_DEFAULT_INTERVAL_MINUTES = 10


def run_improve_monitoring_scheduler() -> None:
    """Scan experiments with connected repos and run improve analysis.

    Called every 10 minutes by the Huey periodic task. Each experiment has its
    own rate-limit check via the ``mlflow.improve.last_monitor_time`` tag.
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
    from mlflow.genai.improve import analyze
    from mlflow.utils.mlflow_tags import (
        MLFLOW_IMPROVE_GITHUB_REPO,
        MLFLOW_IMPROVE_LAST_MONITOR_TIME,
    )

    tags = exp.tags or {}
    repo_url = tags.get(MLFLOW_IMPROVE_GITHUB_REPO)
    if not repo_url:
        return

    if tags.get("mlflow.improve.active_monitor") != "true":
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
