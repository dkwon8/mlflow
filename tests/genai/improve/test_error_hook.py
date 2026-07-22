"""Tests for the event-driven error hook."""

from unittest import mock

import pytest


@mock.patch("mlflow.server.jobs.submit_job")
@mock.patch("mlflow.client.MlflowClient")
def test_submits_job_on_first_error(mock_client_cls, mock_submit):
    from mlflow.genai.improve._error_hook import (
        _last_error_analysis,
        maybe_submit_error_analysis,
    )

    _last_error_analysis.clear()

    exp = mock.MagicMock()
    exp.tags = {"mlflow.improve.github_repo": "owner/repo"}
    mock_client_cls.return_value.get_experiment.return_value = exp

    result = maybe_submit_error_analysis("exp-001")

    assert result is True
    mock_submit.assert_called_once()


@mock.patch("mlflow.server.jobs.submit_job")
@mock.patch("mlflow.client.MlflowClient")
def test_rate_limited_within_cooldown(mock_client_cls, mock_submit):
    from mlflow.genai.improve._error_hook import (
        _last_error_analysis,
        maybe_submit_error_analysis,
    )

    _last_error_analysis.clear()

    exp = mock.MagicMock()
    exp.tags = {"mlflow.improve.github_repo": "owner/repo"}
    mock_client_cls.return_value.get_experiment.return_value = exp

    maybe_submit_error_analysis("exp-002")
    mock_submit.reset_mock()

    result = maybe_submit_error_analysis("exp-002")

    assert result is False
    mock_submit.assert_not_called()


@mock.patch("mlflow.server.jobs.submit_job")
@mock.patch("mlflow.client.MlflowClient")
def test_submits_after_cooldown(mock_client_cls, mock_submit):
    from mlflow.genai.improve._error_hook import (
        _ERROR_COOLDOWN_SECONDS,
        _last_error_analysis,
        maybe_submit_error_analysis,
    )

    _last_error_analysis.clear()

    exp = mock.MagicMock()
    exp.tags = {"mlflow.improve.github_repo": "owner/repo"}
    mock_client_cls.return_value.get_experiment.return_value = exp

    maybe_submit_error_analysis("exp-003")
    mock_submit.reset_mock()

    with mock.patch("mlflow.genai.improve._error_hook.time") as mock_time:
        mock_time.monotonic.return_value = 99999999.0
        result = maybe_submit_error_analysis("exp-003")

    assert result is True
    mock_submit.assert_called_once()


@mock.patch("mlflow.client.MlflowClient")
def test_skips_without_repo(mock_client_cls):
    from mlflow.genai.improve._error_hook import (
        _last_error_analysis,
        maybe_submit_error_analysis,
    )

    _last_error_analysis.clear()

    exp = mock.MagicMock()
    exp.tags = {}
    mock_client_cls.return_value.get_experiment.return_value = exp

    result = maybe_submit_error_analysis("exp-no-repo")

    assert result is False


@mock.patch("mlflow.server.jobs.submit_job", side_effect=RuntimeError("job system down"))
@mock.patch("mlflow.client.MlflowClient")
def test_fails_silently(mock_client_cls, mock_submit):
    from mlflow.genai.improve._error_hook import (
        _last_error_analysis,
        maybe_submit_error_analysis,
    )

    _last_error_analysis.clear()

    exp = mock.MagicMock()
    exp.tags = {"mlflow.improve.github_repo": "owner/repo"}
    mock_client_cls.return_value.get_experiment.return_value = exp

    result = maybe_submit_error_analysis("exp-fail")

    assert result is False
