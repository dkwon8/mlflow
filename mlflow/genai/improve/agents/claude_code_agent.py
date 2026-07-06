"""
Claude Code agent implementation for creating fix PRs.

Clones the repository, runs Claude Code CLI to analyze the issue
and create a fix, then opens a pull request on GitHub.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

from mlflow.genai.improve.code_agent import CodeAgent, FixRequest, FixResult

_logger = logging.getLogger(__name__)


class ClaudeCodeAgent(CodeAgent):
    """Code agent that uses Claude Code CLI to create fix PRs."""

    def name(self) -> str:
        return "claude-code"

    def create_fix(self, request: FixRequest) -> FixResult:
        cli_cmd = shutil.which("claude")
        if not cli_cmd:
            return FixResult(
                success=False,
                error="Claude Code CLI not found. Install it: https://docs.anthropic.com/en/docs/claude-code",
            )

        tmpdir = tempfile.mkdtemp(prefix="mlflow-improve-")
        try:
            repo_dir = self._clone_repo(request.repo_url, request.branch, tmpdir)
            branch_name = f"improve/fix-{request.issue_id[:12]}"
            self._create_branch(repo_dir, branch_name)

            prompt = self._build_prompt(request)
            result = self._run_claude(cli_cmd, repo_dir, prompt)

            if result.returncode != 0:
                return FixResult(
                    success=False,
                    error=f"Claude Code exited with code {result.returncode}: {result.stderr[:500]}",
                )

            pr_url = self._create_pr(repo_dir, branch_name, request)
            if pr_url:
                return FixResult(
                    success=True,
                    pr_url=pr_url,
                    changes_summary=result.stdout[:1000],
                )
            else:
                return FixResult(
                    success=False,
                    error="Claude Code ran but no changes were made to create a PR.",
                )
        except Exception as e:
            _logger.exception("Failed to create fix PR")
            return FixResult(success=False, error=str(e))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def _clone_repo(self, repo_url: str, branch: str, tmpdir: str) -> Path:
        if not repo_url.startswith("http"):
            repo_url = f"https://github.com/{repo_url}.git"
        elif not repo_url.endswith(".git"):
            repo_url = f"{repo_url}.git"

        repo_dir = Path(tmpdir) / "repo"
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", branch, repo_url, str(repo_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
        return repo_dir

    def _create_branch(self, repo_dir: Path, branch_name: str):
        subprocess.run(
            ["git", "checkout", "-b", branch_name],
            cwd=repo_dir,
            check=True,
            capture_output=True,
        )

    def _build_prompt(self, request: FixRequest) -> str:
        root_causes = "\n".join(f"- {rc}" for rc in request.root_causes if rc)
        return (
            f"Fix the following issue detected by MLflow's improve system.\n\n"
            f"Issue: {request.issue_name}\n\n"
            f"Description: {request.issue_description}\n\n"
            f"Root causes:\n{root_causes}\n\n"
            f"Analyze the codebase, find the source of this issue, and fix it. "
            f"Make minimal, targeted changes. Commit your changes with a clear message."
        )

    def _run_claude(self, cli_cmd: str, repo_dir: Path, prompt: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [cli_cmd, "--print", "--dangerously-skip-permissions", prompt],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=300,
        )

    def _create_pr(self, repo_dir: Path, branch_name: str, request: FixRequest) -> str | None:
        diff = subprocess.run(
            ["git", "diff", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
        )
        if not diff.stdout.strip():
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo_dir,
                capture_output=True,
                text=True,
            )
            if not status.stdout.strip():
                return None

        subprocess.run(
            ["git", "push", "origin", branch_name],
            cwd=repo_dir,
            check=True,
            capture_output=True,
            text=True,
        )

        gh_cmd = shutil.which("gh")
        if not gh_cmd:
            _logger.warning("gh CLI not found — branch pushed but PR not created")
            return None

        result = subprocess.run(
            [
                gh_cmd, "pr", "create",
                "--title", f"[MLflow Improve] Fix: {request.issue_name}",
                "--body", (
                    f"## Auto-generated fix by MLflow Improve\n\n"
                    f"**Issue:** {request.issue_name}\n\n"
                    f"**Description:** {request.issue_description}\n\n"
                    f"**Detected by:** `mlflow.genai.improve.analyze()`\n"
                    f"**Experiment:** {request.experiment_id}\n"
                ),
                "--head", branch_name,
            ],
            cwd=repo_dir,
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            return result.stdout.strip()
        else:
            _logger.warning("Failed to create PR: %s", result.stderr)
            return None
