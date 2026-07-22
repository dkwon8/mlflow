"""Tests for the GitHub API file fetcher."""

import base64
from unittest import mock

import pytest


def test_parse_repo_slug_shorthand():
    from mlflow.genai.improve.github_fetcher import _parse_repo_slug

    assert _parse_repo_slug("owner/repo") == ("owner", "repo")


def test_parse_repo_slug_url():
    from mlflow.genai.improve.github_fetcher import _parse_repo_slug

    assert _parse_repo_slug("https://github.com/owner/repo") == ("owner", "repo")


def test_parse_repo_slug_url_with_git():
    from mlflow.genai.improve.github_fetcher import _parse_repo_slug

    assert _parse_repo_slug("https://github.com/owner/repo.git") == ("owner", "repo")


def test_parse_repo_slug_invalid():
    from mlflow.genai.improve.github_fetcher import _parse_repo_slug

    with pytest.raises(ValueError, match="Cannot parse"):
        _parse_repo_slug("not-a-repo")


def test_get_github_session_with_token():
    from mlflow.genai.improve.github_fetcher import _get_github_session

    with mock.patch.dict("os.environ", {"GITHUB_TOKEN": "ghp_test123"}):
        session = _get_github_session()

    assert session.headers["Authorization"] == "Bearer ghp_test123"


def test_get_github_session_without_token():
    from mlflow.genai.improve.github_fetcher import _get_github_session

    with mock.patch.dict("os.environ", {}, clear=True):
        session = _get_github_session()

    assert "Authorization" not in session.headers


def _make_tree_response(files):
    """Build a mock GitHub Trees API response."""
    tree = []
    for path, size in files:
        tree.append({"path": path, "sha": "abc123", "size": size, "type": "blob"})
    return {"tree": tree, "truncated": False}


def _make_content_response(text):
    """Build a mock GitHub Contents API response."""
    encoded = base64.b64encode(text.encode()).decode()
    return {"content": encoded, "encoding": "base64"}


@mock.patch("mlflow.genai.improve.github_fetcher._get_github_session")
def test_fetch_repo_files_filters_and_scores(mock_session_factory):
    from mlflow.genai.improve.github_fetcher import fetch_repo_files

    session = mock.MagicMock()
    mock_session_factory.return_value = session

    tree_resp = mock.MagicMock()
    tree_resp.json.return_value = _make_tree_response([
        ("agent.py", 500),
        ("image.png", 1000),
        ("node_modules/dep.js", 200),
        ("config.yaml", 100),
        ("README.md", 300),
    ])
    tree_resp.headers = {"X-RateLimit-Remaining": "4999"}

    content_resp = mock.MagicMock()
    content_resp.status_code = 200
    content_resp.json.return_value = _make_content_response("print('hello')")

    session.get.side_effect = [tree_resp] + [content_resp] * 10

    files = fetch_repo_files("owner/repo")

    paths = [p for p, _ in files]
    assert "agent.py" in paths
    assert "config.yaml" in paths
    assert "README.md" in paths
    assert "image.png" not in paths
    assert "node_modules/dep.js" not in paths


@mock.patch("mlflow.genai.improve.github_fetcher._get_github_session")
def test_fetch_repo_files_respects_char_budget(mock_session_factory):
    from mlflow.genai.improve.github_fetcher import fetch_repo_files

    session = mock.MagicMock()
    mock_session_factory.return_value = session

    tree_resp = mock.MagicMock()
    tree_resp.json.return_value = _make_tree_response([
        ("a.py", 100),
        ("b.py", 100),
        ("c.py", 100),
    ])
    tree_resp.headers = {"X-RateLimit-Remaining": "4999"}

    big_content = "x" * 60
    content_resp = mock.MagicMock()
    content_resp.status_code = 200
    content_resp.json.return_value = _make_content_response(big_content)

    session.get.side_effect = [tree_resp] + [content_resp] * 5

    files = fetch_repo_files("owner/repo", max_chars=100)

    total = sum(len(c) for _, c in files)
    assert total <= 100


@mock.patch("mlflow.genai.improve.github_fetcher._get_github_session")
def test_fetch_repo_files_uses_trace_hints(mock_session_factory):
    from mlflow.genai.improve.github_fetcher import fetch_repo_files

    session = mock.MagicMock()
    mock_session_factory.return_value = session

    tree_resp = mock.MagicMock()
    tree_resp.json.return_value = _make_tree_response([
        ("utils.py", 100),
        ("parse_resume.py", 100),
    ])
    tree_resp.headers = {"X-RateLimit-Remaining": "4999"}

    content_resp = mock.MagicMock()
    content_resp.status_code = 200
    content_resp.json.return_value = _make_content_response("code here")

    session.get.side_effect = [tree_resp] + [content_resp] * 5

    files = fetch_repo_files("owner/repo", trace_hints=["parse_resume"])

    paths = [p for p, _ in files]
    assert paths[0] == "parse_resume.py"
