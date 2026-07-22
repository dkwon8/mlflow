"""
Event-driven error hook for the MLflow improve system.

When an ERROR trace is detected (via SDK export or OTLP ingest),
this module decides whether to submit an improve analysis job.
Rate-limits to at most one analysis per experiment per cooldown window.
"""

from __future__ import annotations

import logging
import threading
import time

_logger = logging.getLogger(__name__)

_ERROR_COOLDOWN_SECONDS = 120
_last_error_analysis: dict[str, float] = {}
_lock = threading.Lock()


def maybe_submit_error_analysis(experiment_id: str) -> bool:
    """Submit an improve analysis job if cooldown has elapsed.

    Returns True if a job was submitted, False if skipped.
    Thread-safe.
    """
    now = time.monotonic()

    with _lock:
        last = _last_error_analysis.get(experiment_id, 0)
        if now - last < _ERROR_COOLDOWN_SECONDS:
            _logger.debug(
                "Skipping error-triggered analysis for %s (cooldown: %.0fs remaining)",
                experiment_id,
                _ERROR_COOLDOWN_SECONDS - (now - last),
            )
            return False
        _last_error_analysis[experiment_id] = now

    try:
        from mlflow.client import MlflowClient

        client = MlflowClient()
        exp = client.get_experiment(experiment_id)
        tags = exp.tags or {}
        repo_url = tags.get("mlflow.improve.github_repo")
        if not repo_url:
            return False
    except Exception:
        _logger.debug("Failed to check experiment tags for error hook", exc_info=True)
        return False

    try:
        from mlflow.genai.improve.background_work import invoke_improve_analysis_job
        from mlflow.server.jobs import submit_job

        submit_job(
            function=invoke_improve_analysis_job,
            params={"experiment_id": experiment_id, "trace_count": 10},
        )
        _logger.info(
            "Submitted error-triggered improve analysis for experiment %s",
            experiment_id,
        )
        return True
    except Exception:
        _logger.debug("Failed to submit error-triggered analysis job", exc_info=True)
        return False
