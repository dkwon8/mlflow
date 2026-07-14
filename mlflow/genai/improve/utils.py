from __future__ import annotations

import re


def normalize_repo_url(raw_url: str) -> str | None:
    """Convert a git remote URL to ``owner/repo`` format.

    Supports HTTPS (``https://github.com/owner/repo.git``) and SSH
    (``git@github.com:owner/repo.git``) URLs.  Returns ``None`` for
    non-GitHub URLs so callers can silently skip them.
    """
    if not raw_url:
        return None

    raw_url = raw_url.strip()

    match = re.match(r"https?://github\.com/([^/]+/[^/]+?)(?:\.git)?/?$", raw_url)
    if match:
        return match.group(1)

    match = re.match(r"git@github\.com:([^/]+/[^/]+?)(?:\.git)?$", raw_url)
    if match:
        return match.group(1)

    if re.match(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$", raw_url):
        return raw_url

    return None
