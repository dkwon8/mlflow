"""Tests for the 4 analysis modes in analyze()."""

from unittest import mock

import pytest

from mlflow.genai.improve.trace_analyzer import Finding
from mlflow.genai.improve.code_analyzer import CodeFinding


def _mock_experiment():
    exp = mock.MagicMock()
    exp.experiment_id = "exp-001"
    exp.tags = {}
    return exp


def _mock_traces_df(n=5):
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


@mock.patch("mlflow.genai.improve.clone_or_fetch_repo")
@mock.patch("mlflow.genai.improve.analyze_code")
@mock.patch("mlflow.genai.improve.select_relevant_files")
@mock.patch("mlflow.search_traces")
@mock.patch("mlflow.get_experiment_by_name")
def test_traces_only(mock_exp, mock_search, mock_select, mock_analyze_code, mock_clone):
    from mlflow.genai.improve import analyze

    mock_exp.return_value = _mock_experiment()
    mock_search.return_value = _mock_traces_df(5)

    result = analyze("test-exp", mode="traces_only")

    assert result["summary"]["status"] == "ok"
    mock_clone.assert_not_called()
    mock_analyze_code.assert_not_called()


@mock.patch("mlflow.genai.improve.clone_or_fetch_repo")
@mock.patch("mlflow.genai.improve.analyze_code")
@mock.patch("mlflow.genai.improve.select_relevant_files")
@mock.patch("mlflow.search_traces")
@mock.patch("mlflow.get_experiment_by_name")
def test_code_only(mock_exp, mock_search, mock_select, mock_analyze_code, mock_clone):
    from mlflow.genai.improve import analyze

    exp = _mock_experiment()
    mock_exp.return_value = exp
    mock_clone.return_value = "/tmp/repo"
    mock_select.return_value = [("main.py", "print('hello')")]
    mock_analyze_code.return_value = [
        CodeFinding(
            pattern="config_issue",
            severity="medium",
            description="hardcoded value",
            file_path="main.py",
        ),
    ]

    result = analyze("test-exp", mode="code_only", repo_url="owner/repo")

    assert len(result["code_findings"]) == 1
    mock_search.assert_not_called()


@mock.patch("mlflow.genai.improve.clone_or_fetch_repo")
@mock.patch("mlflow.genai.improve.analyze_code")
@mock.patch("mlflow.genai.improve.select_relevant_files")
@mock.patch("mlflow.search_traces")
@mock.patch("mlflow.get_experiment_by_name")
def test_both_mode(mock_exp, mock_search, mock_select, mock_analyze_code, mock_clone):
    from mlflow.genai.improve import analyze

    mock_exp.return_value = _mock_experiment()
    mock_search.return_value = _mock_traces_df(5)
    mock_clone.return_value = "/tmp/repo"
    mock_select.return_value = [("main.py", "code")]
    mock_analyze_code.return_value = [
        CodeFinding(pattern="security", severity="high", description="exposed key"),
    ]

    result = analyze("test-exp", mode="both", repo_url="owner/repo")

    assert result["summary"]["repo_analyzed"] is True
    assert len(result["code_findings"]) == 1


@mock.patch("mlflow.genai.improve.clone_or_fetch_repo")
@mock.patch("mlflow.genai.improve.analyze_code")
@mock.patch("mlflow.genai.improve.select_relevant_files")
@mock.patch("mlflow.search_traces")
@mock.patch("mlflow.get_experiment_by_name")
def test_auto_with_traces(mock_exp, mock_search, mock_select, mock_analyze_code, mock_clone):
    from mlflow.genai.improve import analyze

    mock_exp.return_value = _mock_experiment()
    mock_search.return_value = _mock_traces_df(5)
    mock_clone.return_value = "/tmp/repo"
    mock_select.return_value = []
    mock_analyze_code.return_value = []

    result = analyze("test-exp", mode="auto", repo_url="owner/repo")

    assert result["summary"]["status"] == "ok"
    assert result["summary"]["traces_analyzed"] == 5


@mock.patch("mlflow.search_traces")
@mock.patch("mlflow.get_experiment_by_name")
def test_auto_no_traces_no_repo(mock_exp, mock_search):
    from mlflow.genai.improve import analyze
    import pandas as pd

    mock_exp.return_value = _mock_experiment()
    mock_search.return_value = pd.DataFrame()

    result = analyze("test-exp", mode="auto")

    assert result["summary"]["traces_analyzed"] == 0


@mock.patch("mlflow.get_experiment_by_name")
def test_no_experiment(mock_exp):
    from mlflow.genai.improve import analyze

    mock_exp.return_value = None

    result = analyze("nonexistent")

    assert result["summary"]["status"] == "no_experiment"
    assert result["findings"] == []
