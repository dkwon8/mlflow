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
            [cli_cmd, "-p", prompt, "--dangerously-skip-permissions"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=300,
        )

    def _create_pr(self, repo_dir: Path, branch_name: str, request: FixRequest) -> str | None:
        # Check for committed changes on this branch vs origin
        diff = subprocess.run(
            ["git", "log", "origin/" + request.branch + "..HEAD", "--oneline"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
        )
        # Also check for uncommitted changes
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
        )
        if not diff.stdout.strip() and not status.stdout.strip():
            return None

        # Stage and commit any uncommitted changes
        if status.stdout.strip():
            subprocess.run(["git", "add", "-A"], cwd=repo_dir, check=True, capture_output=True)
            subprocess.run(
                ["git", "commit", "-m", f"fix: {request.issue_name}"],
                cwd=repo_dir, capture_output=True, text=True,
            )

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

        body = self._build_pr_body(request)
        result = subprocess.run(
            [
                gh_cmd, "pr", "create",
                "--title", f"[MLflow Improve] Fix: {request.issue_name}",
                "--body", body,
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

    def _build_pr_body(self, request: FixRequest) -> str:
        sections = [f"## Summary\n\n- Fixed: {request.issue_name}\n- {request.issue_description}"]

        if request.root_causes:
            causes = "\n".join(f"- {rc}" for rc in request.root_causes if rc)
            sections.append(f"## Root Cause\n\n{causes}")

        if request.trace_id or request.failing_span:
            ref_lines = []
            if request.trace_id:
                ref_lines.append(f"- Trace ID: `{request.trace_id}`")
            if request.failing_span:
                ref_lines.append(f"- Failing span: `{request.failing_span}`")
            if request.error_message:
                ref_lines.append(f"- Error: `{request.error_message[:200]}`")
            sections.append(f"## Trace Reference\n\n" + "\n".join(ref_lines))

        sections.append(
            "## Test Plan\n\n"
            "- [ ] Verify the fix resolves the original error\n"
            "- [ ] Confirm MLflow traces show successful completions\n"
            "- [ ] Check no regression in related functionality"
        )

        sections.append(
            "---\n"
            "Generated with [Claude Code](https://docs.anthropic.com/en/docs/claude-code) "
            "via `mlflow.genai.improve`"
        )

        return "\n\n".join(sections)
