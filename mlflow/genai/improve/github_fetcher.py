"""
GitHub API-based file fetcher for the MLflow improve system.

Fetches repository files via the GitHub REST API instead of cloning
to disk. Produces the same list[tuple[str, str]] output that
analyze_code() expects, making it a drop-in replacement for
clone_or_fetch_repo() + select_relevant_files().

Requires GITHUB_TOKEN environment variable for authentication.
"""

from __future__ import annotations

import base64
import logging
import os
import re

import requests

from .code_analyzer import (
    _MAX_CHAR_BUDGET,
    _RELEVANT_EXTENSIONS,
    _SKIP_DIRS,
    _compute_priority,
)

_logger = logging.getLogger(__name__)

_GITHUB_API_BASE = "https://api.github.com"


def _parse_repo_slug(repo_url: str) -> tuple[str, str]:
    """Extract owner and repo name from a URL or owner/repo shorthand."""
    repo_url = repo_url.rstrip("/")
    if repo_url.endswith(".git"):
        repo_url = repo_url[:-4]

    match = re.match(r"https?://github\.com/([^/]+)/([^/]+)", repo_url)
    if match:
        return match.group(1), match.group(2)

    if "/" in repo_url and not repo_url.startswith("http"):
        parts = repo_url.split("/")
        if len(parts) == 2:
            return parts[0], parts[1]

    raise ValueError(
        f"Cannot parse GitHub repo from '{repo_url}'. "
        "Use 'owner/repo' format or a full GitHub URL."
    )


def _resolve_github_token() -> str | None:
    """Resolve a GitHub token from environment or the gh CLI."""
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        return token

    import subprocess
    try:
        result = subprocess.run(
            ["gh", "auth", "token"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return None


def _get_github_session() -> requests.Session:
    """Create a requests Session with GitHub API headers.

    Token resolution order:
    1. GITHUB_TOKEN environment variable
    2. ``gh auth token`` (GitHub CLI keyring)
    3. Unauthenticated (60 req/hr, public repos only)
    """
    session = requests.Session()
    session.headers.update({
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    })

    token = _resolve_github_token()
    if token:
        session.headers["Authorization"] = f"Bearer {token}"
    else:
        _logger.warning(
            "No GitHub token found — using unauthenticated API (60 req/hr limit). "
            "Either set GITHUB_TOKEN or install the GitHub CLI (gh auth login)."
        )

    return session


def _fetch_file_tree(
    session: requests.Session,
    owner: str,
    repo: str,
    branch: str,
) -> list[dict]:
    """Fetch the full recursive file tree via the GitHub Trees API.

    Returns a list of dicts with keys: path, sha, size, type.
    One API call for the entire tree.
    """
    url = f"{_GITHUB_API_BASE}/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
    resp = session.get(url, timeout=30)
    resp.raise_for_status()

    data = resp.json()
    if data.get("truncated"):
        _logger.warning(
            "GitHub tree response was truncated for %s/%s — "
            "some files may be missing from analysis",
            owner, repo,
        )

    remaining = resp.headers.get("X-RateLimit-Remaining")
    if remaining and int(remaining) < 100:
        _logger.warning(
            "GitHub API rate limit low: %s requests remaining", remaining
        )

    return data.get("tree", [])


def _fetch_file_content(
    session: requests.Session,
    owner: str,
    repo: str,
    path: str,
    ref: str,
) -> str | None:
    """Fetch a single file's decoded text content via the GitHub Contents API."""
    url = f"{_GITHUB_API_BASE}/repos/{owner}/{repo}/contents/{path}"
    resp = session.get(url, params={"ref": ref}, timeout=15)

    if resp.status_code == 404:
        return None
    resp.raise_for_status()

    data = resp.json()
    encoding = data.get("encoding", "")
    content = data.get("content", "")

    if encoding == "base64" and content:
        try:
            return base64.b64decode(content).decode("utf-8", errors="replace")
        except Exception:
            _logger.debug("Failed to decode %s", path)
            return None

    return None


def fetch_repo_files(
    repo_slug: str,
    branch: str = "main",
    trace_hints: list[str] | None = None,
    max_chars: int = _MAX_CHAR_BUDGET,
) -> list[tuple[str, str]]:
    """Fetch relevant files from a GitHub repo via REST API.

    Drop-in replacement for clone_or_fetch_repo() + select_relevant_files().
    Always fetches the latest code — no cache, no disk, no stale state.

    Args:
        repo_slug: GitHub repo in "owner/repo" format or full URL.
        branch: Branch to fetch from (default "main").
        trace_hints: Tool/span names from traces to guide file selection.
        max_chars: Maximum total characters to fetch.

    Returns:
        List of (relative_path, file_content) tuples — same format
        as select_relevant_files().
    """
    owner, repo = _parse_repo_slug(repo_slug)
    session = _get_github_session()

    tree = _fetch_file_tree(session, owner, repo, branch)

    candidates = []
    for entry in tree:
        if entry.get("type") != "blob":
            continue

        path = entry["path"]
        size = entry.get("size", 0)

        if size > 500_000 or size == 0:
            continue

        parts = path.split("/")
        if any(d in _SKIP_DIRS for d in parts[:-1]):
            continue

        fname = parts[-1]
        ext = ""
        if "." in fname:
            ext = "." + fname.rsplit(".", 1)[1].lower()
        if ext not in _RELEVANT_EXTENSIONS:
            continue

        priority = _compute_priority(path, fname, ext, trace_hints)
        candidates.append((priority, size, path))

    candidates.sort(key=lambda c: (-c[0], c[1]))

    selected: list[tuple[str, str]] = []
    total_chars = 0

    for priority, size, path in candidates:
        if total_chars + size > max_chars:
            continue

        content = _fetch_file_content(session, owner, repo, path, branch)
        if content is None:
            continue

        selected.append((path, content))
        total_chars += len(content)

    _logger.info(
        "Fetched %d files (%d chars) from %s/%s via GitHub API",
        len(selected), total_chars, owner, repo,
    )
    return selected
