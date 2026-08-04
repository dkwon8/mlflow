"""
Dynamic code analyzer for the MLflow improve system.

Clones a GitHub repository, intelligently selects relevant files,
and uses an LLM to analyze the code for issues — independent of traces.
Works with any agent codebase, not just MLflow-specific patterns.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import pydantic

_logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "openai:/gpt-5.4-mini"

_RELEVANT_EXTENSIONS = {
    ".py", ".ts", ".js", ".tsx", ".jsx",
    ".yaml", ".yml", ".toml", ".json",
    ".md",
}

_PRIORITY_KEYWORDS = {
    "agent", "prompt", "system", "config", "tool",
    "handler", "pipeline", "chain", "llm", "model",
    "server", "mcp", "scoring", "filter", "judge",
}

_SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    "dist", "build", ".next", ".cache", "env",
}

_MAX_CHAR_BUDGET = 150_000


@dataclass
class CodeFinding:
    """A code-level issue detected by LLM analysis."""
    pattern: str
    severity: str  # "low", "medium", "high"
    description: str
    file_path: str | None = None
    root_cause: str | None = None
    suggested_fix: str | None = None
    confidence: float = 0.0
    evidence: dict = field(default_factory=dict)


class _CodeIssue(pydantic.BaseModel):
    """Structured output schema for LLM code analysis."""
    category: str = pydantic.Field(
        description="Issue category: anti_pattern, prompt_quality, config_issue, "
        "missing_error_handling, context_management, tool_schema, performance, security"
    )
    severity: str = pydantic.Field(
        description="low, medium, or high",
        default="medium",
    )
    file: str = pydantic.Field(description="Relative path to the affected file")
    problem: str = pydantic.Field(description="What the issue is")
    why: str = pydantic.Field(description="Why this is a problem")
    fix: str = pydantic.Field(description="Specific code change to fix it")
    confidence: float = pydantic.Field(description="0.0 to 1.0", ge=0.0, le=1.0)


class _AnalysisResponse(pydantic.BaseModel):
    """Top-level structured output for code analysis."""
    issues: list[_CodeIssue]
    summary: str = pydantic.Field(
        description="Brief overall assessment of code quality",
        default="",
    )


_repo_cache: dict[str, Path] = {}


def clone_or_fetch_repo(repo_url: str, branch: str = "main") -> Path:
    """Clone a GitHub repo to a temp directory, with session-level caching.

    Args:
        repo_url: GitHub repo URL or owner/repo shorthand.
        branch: Branch to clone.

    Returns:
        Path to the cloned repo directory.
    """
    cache_key = f"{repo_url}@{branch}"

    if cache_key in _repo_cache:
        cached = _repo_cache[cache_key]
        if cached.exists():
            _logger.debug("Refreshing cached repo at %s", cached)
            subprocess.run(
                ["git", "fetch", "origin", branch],
                cwd=cached, capture_output=True, text=True, timeout=60,
            )
            subprocess.run(
                ["git", "reset", "--hard", f"origin/{branch}"],
                cwd=cached, capture_output=True, text=True, timeout=30,
            )
            return cached
        del _repo_cache[cache_key]

    normalized_url = repo_url
    if not normalized_url.startswith("http"):
        normalized_url = f"https://github.com/{normalized_url}.git"
    elif not normalized_url.endswith(".git"):
        normalized_url = f"{normalized_url}.git"

    tmpdir = tempfile.mkdtemp(prefix="mlflow-improve-")
    repo_dir = Path(tmpdir) / "repo"

    _logger.info("Cloning %s (branch: %s)", normalized_url, branch)
    subprocess.run(
        ["git", "clone", "--single-branch", "--branch", branch,
         normalized_url, str(repo_dir)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )

    _repo_cache[cache_key] = repo_dir
    return repo_dir


def _compute_priority(
    rel_path: str,
    filename: str,
    ext: str,
    trace_hints: list[str] | None,
) -> int:
    """Compute a priority score for a file (higher = more relevant)."""
    score = 0
    path_lower = rel_path.lower()
    name_lower = filename.lower().rsplit(".", 1)[0]

    for keyword in _PRIORITY_KEYWORDS:
        if keyword in path_lower:
            score += 10
        if keyword in name_lower:
            score += 5

    if ext == ".py":
        score += 3
    elif ext in (".ts", ".tsx", ".js", ".jsx"):
        score += 2

    if name_lower in ("readme", "requirements", "pyproject", "package"):
        score += 8

    if name_lower.startswith("test_") or name_lower.endswith("_test"):
        score -= 5

    if trace_hints:
        for hint in trace_hints:
            hint_lower = hint.lower().replace(" ", "_").replace("-", "_")
            if hint_lower in path_lower:
                score += 15

    depth = rel_path.count("/")
    score -= depth

    return score


def analyze_code(
    selected_files: list[tuple[str, str]],
    trace_findings: list[dict] | None = None,
    model: str = _DEFAULT_MODEL,
) -> list[CodeFinding]:
    """Use an LLM to analyze code and find issues.

    Args:
        selected_files: List of (relative_path, content) tuples.
        trace_findings: Optional trace-based findings for correlation.
        model: Model URI for the LLM call (default: openai:/gpt-5.4-mini).

    Returns:
        List of CodeFinding objects.
    """
    if not selected_files:
        return []

    from mlflow.genai.utils.llm_utils import _call_llm

    messages = _build_analysis_messages(selected_files, trace_findings)

    try:
        response = _call_llm(
            model=model,
            messages=messages,
            response_format=_AnalysisResponse,
            max_tokens=4096,
        )
    except Exception:
        _logger.exception("LLM code analysis failed")
        return []

    if isinstance(response, _AnalysisResponse):
        result = response
    elif isinstance(response, dict):
        result = _AnalysisResponse.model_validate(response)
    elif isinstance(response, str):
        import json
        result = _AnalysisResponse.model_validate(json.loads(response))
    else:
        content = _extract_content(response)
        if isinstance(content, str):
            import json
            result = _AnalysisResponse.model_validate(json.loads(content))
        else:
            result = _AnalysisResponse.model_validate(content)

    return [
        CodeFinding(
            pattern=issue.category,
            severity=issue.severity,
            description=issue.problem,
            file_path=issue.file,
            root_cause=issue.why,
            suggested_fix=issue.fix,
            confidence=issue.confidence,
            evidence={"source": "code_analysis", "model": model},
        )
        for issue in result.issues
    ]


def _extract_content(response) -> str:
    """Extract text content from various LLM response formats."""
    if hasattr(response, "choices") and response.choices:
        msg = response.choices[0].message
        if hasattr(msg, "parsed") and msg.parsed:
            return msg.parsed
        return msg.content
    if hasattr(response, "content"):
        return response.content
    return str(response)


def _build_analysis_messages(
    selected_files: list[tuple[str, str]],
    trace_findings: list[dict] | None,
) -> list[dict[str, str]]:
    """Build the system + user messages for LLM code analysis."""
    system_prompt = (
        "You are an expert code reviewer specializing in AI agent systems. "
        "Analyze the provided codebase and identify issues that affect reliability, "
        "performance, and correctness.\n\n"
        "Look for these categories of issues:\n"
        "- **anti_pattern**: Bad patterns in agent/LLM code (unbounded loops, missing retries, "
        "hardcoded values that should be configurable)\n"
        "- **prompt_quality**: Vague instructions, missing constraints, conflicting directives "
        "in system prompts or LLM prompts\n"
        "- **config_issue**: Hardcoded API keys, missing timeouts, no rate limiting, "
        "insecure defaults\n"
        "- **missing_error_handling**: Missing try/except around API calls, no fallback "
        "for external service failures\n"
        "- **context_management**: No conversation history truncation, growing context that "
        "will hit token limits at scale\n"
        "- **tool_schema**: Missing or incorrect tool descriptions, parameter schemas that "
        "will confuse the LLM\n"
        "- **performance**: Synchronous calls where async is possible, unnecessary serial "
        "processing, redundant API calls\n"
        "- **security**: Exposed credentials, command injection risks, unsafe file operations\n\n"
        "For each issue, return a JSON object with these exact fields:\n"
        "- category: one of anti_pattern, prompt_quality, config_issue, "
        "missing_error_handling, context_management, tool_schema, performance, security\n"
        "- severity: low, medium, or high\n"
        "- file: relative path to the affected file\n"
        "- problem: what the issue is\n"
        "- why: why it's a problem (root cause)\n"
        "- fix: a concrete code change to fix it (not vague advice)\n"
        "- confidence: 0.0 to 1.0 based on how certain you are\n\n"
        "Return JSON with keys: issues (array of objects), summary (string).\n\n"
        "Focus on real, actionable issues. Do not flag style preferences or minor "
        "nitpicks. Only report issues with confidence >= 0.5."
    )

    file_sections = []
    for rel_path, content in selected_files:
        file_sections.append(f"### {rel_path}\n```\n{content}\n```")

    user_parts = ["# Codebase Files\n\n" + "\n\n".join(file_sections)]

    if trace_findings:
        trace_section = "\n# Runtime Trace Findings\n\n"
        trace_section += (
            "The following issues were detected from runtime traces. "
            "Use these to correlate with code-level root causes:\n\n"
        )
        for f in trace_findings:
            trace_section += (
                f"- **{f.get('pattern', 'unknown')}** ({f.get('severity', '')}): "
                f"{f.get('description', '')}\n"
            )
        user_parts.append(trace_section)

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


