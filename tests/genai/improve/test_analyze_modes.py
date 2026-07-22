"""Tests for analyze() after the dynamic improve refactor."""

from unittest import mock

import pytest

from mlflow.genai.improve.code_analyzer import CodeFinding


def _mock_experiment(exp_id="exp-001", tags=None):
    exp = mock.MagicMock()
    exp.experiment_id = exp_id
    exp.tags = tags or {}
    return exp


def _mock_traces_df(n=12):
    import pandas as pd

    rows = []
    for i in range(n):
        rows.append({
            "trace_id": f"tr-{i:03d}",
            "spans": [],
            "execution_duration": 30_000,
            "assessments": [],
        })
    return pd.DataFrame(rows)


@mock.patch("mlflow.genai.improve.fetch_repo_files")
@mock.patch("mlflow.genai.improve.analyze_code")
@mock.patch("mlflow.search_experiments")
@mock.patch("mlflow.search_traces")
@mock.patch("mlflow.get_experiment_by_name")
def test_analyze_unified(mock_exp, mock_search, mock_search_exps, mock_analyze_code, mock_fetch):
    from mlflow.genai.improve import analyze

    mock_exp.return_value = _mock_experiment()
    mock_search.return_value = _mock_traces_df(12)
    mock_search_exps.return_value = [_mock_experiment()]
    mock_fetch.return_value = [("main.py", "print('hello')")]
    mock_analyze_code.return_value = []

    result = analyze("test-exp", repo_url="owner/repo")

    assert result["summary"]["status"] == "ok"
    assert result["summary"]["traces_analyzed"] == 12
    mock_fetch.assert_called_once()
    mock_analyze_code.assert_called_once()


@mock.patch("mlflow.genai.improve.fetch_repo_files")
@mock.patch("mlflow.genai.improve.analyze_code")
@mock.patch("mlflow.search_experiments")
@mock.patch("mlflow.search_traces")
@mock.patch("mlflow.get_experiment_by_name")
def test_analyze_with_code_findings(mock_exp, mock_search, mock_search_exps, mock_analyze_code, mock_fetch):
    from mlflow.genai.improve import analyze

    mock_exp.return_value = _mock_experiment()
    mock_search.return_value = _mock_traces_df(12)
    mock_search_exps.return_value = [_mock_experiment()]
    mock_fetch.return_value = [("main.py", "code")]
    mock_analyze_code.return_value = [
        CodeFinding(pattern="security", severity="high", description="exposed key"),
    ]

    result = analyze("test-exp", repo_url="owner/repo")

    assert len(result["code_findings"]) == 1
    assert result["code_findings"][0]["pattern"] == "security"


@mock.patch("mlflow.get_experiment_by_name")
def test_no_experiment(mock_exp):
    from mlflow.genai.improve import analyze

    mock_exp.return_value = None

    result = analyze("nonexistent")

    assert result["summary"]["status"] == "no_experiment"
    assert result["findings"] == []


@mock.patch("mlflow.genai.improve.fetch_repo_files")
@mock.patch("mlflow.genai.improve.analyze_code")
@mock.patch("mlflow.search_experiments")
@mock.patch("mlflow.search_traces")
@mock.patch("mlflow.get_experiment_by_name")
def test_no_repo(mock_exp, mock_search, mock_search_exps, mock_analyze_code, mock_fetch):
    from mlflow.genai.improve import analyze

    mock_exp.return_value = _mock_experiment()

    result = analyze("test-exp")

    assert result["summary"]["status"] == "no_repo"
    mock_fetch.assert_not_called()


@mock.patch("mlflow.genai.improve.fetch_repo_files")
@mock.patch("mlflow.genai.improve.analyze_code")
@mock.patch("mlflow.search_experiments")
@mock.patch("mlflow.search_traces")
@mock.patch("mlflow.get_experiment_by_name")
def test_insufficient_traces(mock_exp, mock_search, mock_search_exps, mock_analyze_code, mock_fetch):
    from mlflow.genai.improve import analyze

    mock_exp.return_value = _mock_experiment()
    mock_search.return_value = _mock_traces_df(3)
    mock_search_exps.return_value = [_mock_experiment()]

    result = analyze("test-exp", repo_url="owner/repo")

    assert result["summary"]["status"] == "insufficient_traces"


@mock.patch("mlflow.genai.improve.fetch_repo_files")
@mock.patch("mlflow.genai.improve.analyze_code")
@mock.patch("mlflow.search_experiments")
@mock.patch("mlflow.search_traces")
@mock.patch("mlflow.get_experiment_by_name")
def test_cross_experiment_pooling(mock_exp, mock_search, mock_search_exps, mock_analyze_code, mock_fetch):
    from mlflow.genai.improve import analyze

    exp_a = _mock_experiment("exp-A")
    exp_b = _mock_experiment("exp-B")
    mock_exp.return_value = exp_a
    mock_search_exps.return_value = [exp_a, exp_b]
    mock_search.return_value = _mock_traces_df(12)
    mock_fetch.return_value = []
    mock_analyze_code.return_value = []

    result = analyze("test-exp", repo_url="owner/repo")

    call_args = mock_search.call_args
    exp_ids = call_args[1].get("experiment_ids") or call_args[0][0]
    assert "exp-A" in exp_ids
    assert "exp-B" in exp_ids
    assert result["summary"]["experiments_pooled"] == 2


@mock.patch("mlflow.genai.improve.fetch_repo_files")
@mock.patch("mlflow.genai.improve.analyze_code")
@mock.patch("mlflow.search_experiments")
@mock.patch("mlflow.search_traces")
@mock.patch("mlflow.get_experiment_by_name")
def test_pooling_includes_current_experiment(mock_exp, mock_search, mock_search_exps, mock_analyze_code, mock_fetch):
    """Even if search_experiments doesn't return the current experiment, it should be included."""
    from mlflow.genai.improve import analyze

    exp_current = _mock_experiment("exp-current")
    exp_other = _mock_experiment("exp-other")
    mock_exp.return_value = exp_current
    mock_search_exps.return_value = [exp_other]
    mock_search.return_value = _mock_traces_df(12)
    mock_fetch.return_value = []
    mock_analyze_code.return_value = []

    result = analyze("test-exp", repo_url="owner/repo")

    call_args = mock_search.call_args
    exp_ids = call_args[1].get("experiment_ids") or call_args[0][0]
    assert "exp-current" in exp_ids
    assert "exp-other" in exp_ids


@mock.patch("mlflow.genai.improve.fetch_repo_files")
@mock.patch("mlflow.genai.improve.analyze_code")
@mock.patch("mlflow.search_experiments")
@mock.patch("mlflow.search_traces")
@mock.patch("mlflow.get_experiment_by_name")
def test_github_api_failure_degrades_gracefully(mock_exp, mock_search, mock_search_exps, mock_analyze_code, mock_fetch):
    from mlflow.genai.improve import analyze

    mock_exp.return_value = _mock_experiment()
    mock_search.return_value = _mock_traces_df(12)
    mock_search_exps.return_value = [_mock_experiment()]
    mock_fetch.side_effect = ValueError("GITHUB_TOKEN not set")

    result = analyze("test-exp", repo_url="owner/repo")

    assert result["summary"]["status"] == "ok"
    assert result["code_findings"] == []
    assert result["findings"] is not None
