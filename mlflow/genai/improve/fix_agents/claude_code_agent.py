"""
Claude Code agent implementation for creating fix PRs.

Uses the Claude Agent SDK (claude-agent-sdk) for dynamic code analysis
and fix generation. Falls back to the Claude CLI if the SDK is not installed.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

from mlflow.genai.improve.fix_agent_registry import CodeAgent, FixRequest, FixResult
from mlflow.genai.improve.code_analyzer import clone_or_fetch_repo

_logger = logging.getLogger(__name__)


def _sdk_available() -> bool:
    try:
        import claude_agent_sdk  # noqa: F401
        return True
    except ImportError:
        return False


class ClaudeCodeAgent(CodeAgent):
    """Code agent that uses Claude Agent SDK to analyze code and create fix PRs.

    Dynamically reads the repository, reasons about the issue, edits files,
    and opens a pull request — all through the SDK's built-in capabilities.
    Falls back to the Claude CLI (claude -p) if the SDK is not installed.
    """

    def name(self) -> str:
        return "claude-code"

    def create_fix(self, request: FixRequest) -> FixResult:
        try:
            cached_dir = clone_or_fetch_repo(request.repo_url, request.branch)
            work_dir = Path(tempfile.mkdtemp(prefix="mlflow-fix-"))
            shutil.copytree(cached_dir, work_dir / "repo", dirs_exist_ok=True)
            repo_dir = work_dir / "repo"
            branch_name = f"improve/fix-{request.issue_id[:12]}"
            self._create_branch(repo_dir, branch_name)

            prompt = self._build_prompt(request)

            if _sdk_available():
                _logger.info("Using Claude Agent SDK for fix")
                result_text = asyncio.run(
                    self._run_sdk(repo_dir, prompt)
                )
            else:
                _logger.info("Claude Agent SDK not available, falling back to CLI")
                cli_result = self._run_cli(repo_dir, prompt)
                if cli_result is None:
                    return FixResult(
                        success=False,
                        error="Claude Code CLI not found. Install claude-agent-sdk or the Claude CLI.",
                    )
                if cli_result.returncode != 0:
                    return FixResult(
                        success=False,
                        error=f"Claude Code exited with code {cli_result.returncode}: {cli_result.stderr[:500]}",
                    )
                result_text = cli_result.stdout[:1000]

            pr_url = self._create_pr(repo_dir, branch_name, request)
            if pr_url:
                return FixResult(
                    success=True,
                    pr_url=pr_url,
                    changes_summary=result_text[:1000] if result_text else None,
                )
            else:
                return FixResult(
                    success=False,
                    error="Agent ran but no changes were made to create a PR.",
                )
        except Exception as e:
            _logger.exception("Failed to create fix PR")
            return FixResult(success=False, error=str(e))

    async def _run_sdk(self, repo_dir: Path, prompt: str) -> str:
        """Run the Claude Agent SDK to analyze and fix code."""
        from claude_agent_sdk import ClaudeSDKClient

        try:
            import mlflow.anthropic
            mlflow.anthropic.autolog()
        except Exception:
            pass

        messages = []
        async with ClaudeSDKClient(cwd=str(repo_dir)) as client:
            await client.query(prompt)

            async for message in client.receive_response():
                messages.append(message)

        text_parts = []
        for msg in messages:
            if hasattr(msg, "content"):
                content = msg.content
                if isinstance(content, str):
                    text_parts.append(content)
                elif isinstance(content, list):
                    for block in content:
                        if hasattr(block, "text"):
                            text_parts.append(block.text)

        return "\n".join(text_parts)

    def _run_cli(self, repo_dir: Path, prompt: str) -> subprocess.CompletedProcess | None:
        """Fallback: run Claude Code CLI."""
        cli_cmd = shutil.which("claude")
        if not cli_cmd:
            return None
        return subprocess.run(
            [cli_cmd, "-p", prompt, "--dangerously-skip-permissions"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=300,
        )

    def _create_branch(self, repo_dir: Path, branch_name: str):
        subprocess.run(
            ["git", "checkout", "-b", branch_name],
            cwd=repo_dir,
            check=True,
            capture_output=True,
        )

    def _build_prompt(self, request: FixRequest) -> str:
        root_causes = "\n".join(f"- {rc}" for rc in request.root_causes if rc)

        parts = [
            f"Fix the following issue detected by MLflow's improve system.\n",
            f"Issue: {request.issue_name}\n",
            f"Description: {request.issue_description}\n",
            f"Root causes:\n{root_causes}\n",
        ]

        if request.code_findings:
            parts.append("Code analysis findings:\n")
            for cf in request.code_findings[:5]:
                parts.append(
                    f"- [{cf.get('severity', '')}] {cf.get('description', '')} "
                    f"in {cf.get('file_path', 'unknown')}\n"
                    f"  Fix: {cf.get('suggested_fix', 'N/A')}\n"
                )

        parts.append(
            "\nAnalyze the codebase, find the source of this issue, and fix it. "
            "Make minimal, targeted changes. Commit your changes with a clear message."
        )

        return "\n".join(parts)

    def _create_pr(self, repo_dir: Path, branch_name: str, request: FixRequest) -> str | None:
        diff = subprocess.run(
            ["git", "log", "origin/" + request.branch + "..HEAD", "--oneline"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
        )
        if not diff.stdout.strip() and not status.stdout.strip():
            return None

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
            sections.append("## Trace Reference\n\n" + "\n".join(ref_lines))

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
