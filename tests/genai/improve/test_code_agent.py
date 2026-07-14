"""Tests for the Claude code agent."""

from pathlib import Path
from unittest import mock

from mlflow.genai.improve.fix_agents.claude_code_agent import ClaudeCodeAgent
from mlflow.genai.improve.fix_agent_registry import FixRequest


def _make_request(**overrides) -> FixRequest:
    defaults = {
        "issue_id": "s-abc12345",
        "issue_name": "Missing error handling",
        "issue_description": "No try/except around API calls",
        "root_causes": ["External API can fail", "No retry logic"],
        "repo_url": "owner/repo",
        "branch": "main",
        "experiment_id": "exp-001",
    }
    defaults.update(overrides)
    return FixRequest(**defaults)


def test_build_prompt_includes_issue():
    agent = ClaudeCodeAgent()
    request = _make_request()
    prompt = agent._build_prompt(request)

    assert "Missing error handling" in prompt
    assert "No try/except around API calls" in prompt
    assert "External API can fail" in prompt


def test_build_prompt_includes_code_findings():
    agent = ClaudeCodeAgent()
    request = _make_request(code_findings=[
        {"severity": "high", "description": "hardcoded key", "file_path": "config.py", "suggested_fix": "use env var"},
    ])
    prompt = agent._build_prompt(request)

    assert "hardcoded key" in prompt
    assert "config.py" in prompt
    assert "use env var" in prompt


@mock.patch("mlflow.genai.improve.fix_agents.claude_code_agent.clone_or_fetch_repo")
def test_create_fix_uses_isolated_copy(mock_clone, tmp_path):
    mock_clone.return_value = tmp_path / "cached"
    (tmp_path / "cached").mkdir()
    (tmp_path / "cached" / ".git").mkdir()
    (tmp_path / "cached" / "file.py").write_text("code")

    agent = ClaudeCodeAgent()
    request = _make_request()

    with mock.patch.object(agent, "_create_branch"), \
         mock.patch.object(agent, "_run_cli", return_value=mock.MagicMock(returncode=0, stdout="fixed", stderr="")), \
         mock.patch.object(agent, "_create_pr", return_value="https://github.com/pr/1"), \
         mock.patch("mlflow.genai.improve.fix_agents.claude_code_agent._sdk_available", return_value=False):
        result = agent.create_fix(request)

    mock_clone.assert_called_once()
    assert result.success


@mock.patch("shutil.which", return_value=None)
@mock.patch("mlflow.genai.improve.fix_agents.claude_code_agent.clone_or_fetch_repo")
def test_cli_fallback_not_found(mock_clone, mock_which, tmp_path):
    mock_clone.return_value = tmp_path / "cached"
    (tmp_path / "cached").mkdir()
    (tmp_path / "cached" / ".git").mkdir()

    agent = ClaudeCodeAgent()
    request = _make_request()

    with mock.patch("mlflow.genai.improve.fix_agents.claude_code_agent._sdk_available", return_value=False), \
         mock.patch.object(agent, "_create_branch"):
        result = agent.create_fix(request)

    assert not result.success


def test_agent_name():
    agent = ClaudeCodeAgent()
    assert agent.name() == "claude-code"
