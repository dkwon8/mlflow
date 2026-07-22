"""Summary and alert computation for the MLflow improve system.

Takes parsed trace data and findings, computes health stats and
generates deduplicated error alerts.
"""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path

_logger = logging.getLogger(__name__)

_ERROR_TERM_PATTERNS = [
    r"model[_ ]not[_ ]found[:\s]+['\"]?([a-zA-Z0-9._-]+)",
    r"(?:unknown|invalid|unsupported)\s+model[:\s]+['\"]?([a-zA-Z0-9._-]+)",
    r"(?:requested\s+)?model\s+['\"]([a-zA-Z0-9._-]+)['\"]",
    r"(?:KeyError|NameError|AttributeError)[:\s]+['\"]?([a-zA-Z0-9_.]+)",
    r"No module named ['\"]([a-zA-Z0-9_.]+)",
    r"does not exist.*['\"]([a-zA-Z0-9._-]{5,})['\"]",
    r"['\"]([a-zA-Z0-9._-]{5,})['\"].*does not exist",
]

_NOISE_WORDS = {
    "error", "type", "message", "none", "null", "true",
    "false", "code", "status", "data", "value", "result",
}


def compute_alerts(
    parsed_traces: list[dict],
    repo_dir: Path | None = None,
    file_contents: list[tuple[str, str]] | None = None,
) -> list[dict]:
    """Generate deduplicated error alerts from parsed traces.

    Extracts error signatures from the most recent traces, optionally
    checks whether the errors have been fixed in the codebase, and
    returns one alert per unique error signature.

    Pass file_contents (from GitHub API) or repo_dir (from git clone)
    to enable resolved-error filtering.
    """
    recency_window = min(5, len(parsed_traces))
    recent_error_sigs: set[str] = set()
    for p in parsed_traces[:recency_window]:
        for err in p.get("error_details", []):
            sig = f"{err['span_name']}:{(err['error_message'] or '')[:200]}"
            recent_error_sigs.add(sig)

    resolved_in_code: set[str] = set()
    if file_contents:
        resolved_in_code = _check_errors_against_files(recent_error_sigs, file_contents)
    elif repo_dir:
        resolved_in_code = _check_errors_against_code(recent_error_sigs, repo_dir)

    raw_alerts: list[dict] = []
    for p in parsed_traces:
        if p["error_details"]:
            first_error = p["error_details"][0]
            sig = f"{first_error['span_name']}:{(first_error['error_message'] or '')[:200]}"
            if sig not in recent_error_sigs:
                continue
            if sig in resolved_in_code:
                continue

            timestamp = None
            if p.get("start_ns") and p["start_ns"] > 0:
                from datetime import datetime, timezone
                timestamp = datetime.fromtimestamp(
                    p["start_ns"] / 1e9, tz=timezone.utc
                ).isoformat()

            raw_alerts.append({
                "trace_id": p["trace_id"],
                "error_message": first_error["error_message"] or "Unknown error",
                "user_query": p["user_query"] or "",
                "failing_span": first_error["span_name"],
                "severity": "high",
                "timestamp": timestamp,
                "_sig": sig,
            })

    seen_sigs: set[str] = set()
    alerts: list[dict] = []
    for a in raw_alerts:
        if a["_sig"] not in seen_sigs:
            seen_sigs.add(a["_sig"])
            alert = {k: v for k, v in a.items() if k != "_sig"}
            alerts.append(alert)

    return alerts


def compute_summary(
    experiment_name: str,
    parsed_traces: list[dict],
    trace_findings: list[dict],
    code_findings_list: list,
    repo_url: str,
) -> dict:
    """Compute the summary stats dict for an analysis result."""
    total_tool_calls = sum(p["tool_call_count"] for p in parsed_traces)
    avg_tool_calls = total_tool_calls / len(parsed_traces) if parsed_traces else 0
    error_count = sum(1 for p in parsed_traces if p["error_count"] > 0)
    healthy_count = len(parsed_traces) - error_count
    latencies = [p["execution_ms"] for p in parsed_traces if p["execution_ms"] > 0]
    avg_latency_ms = round(sum(latencies) / len(latencies)) if latencies else 0

    return {
        "status": "ok",
        "experiment_name": experiment_name,
        "traces_analyzed": len(parsed_traces),
        "healthy_count": healthy_count,
        "error_count": error_count,
        "avg_latency_ms": avg_latency_ms,
        "findings_count": len(trace_findings) + len(code_findings_list),
        "code_findings_count": len(code_findings_list),
        "avg_tool_calls": round(avg_tool_calls, 1),
        "high_severity": sum(
            1 for f in trace_findings if f.get("severity") == "high"
        ) + sum(
            1 for f in code_findings_list if f.severity == "high"
        ),
        "medium_severity": sum(
            1 for f in trace_findings if f.get("severity") == "medium"
        ) + sum(
            1 for f in code_findings_list if f.severity == "medium"
        ),
        "repo_url": repo_url,
    }


def _check_errors_against_files(
    error_sigs: set[str],
    file_contents: list[tuple[str, str]],
) -> set[str]:
    """Check which error signatures have been fixed, using in-memory file data.

    Same logic as _check_errors_against_code but searches file contents
    directly instead of running grep subprocess on a cloned directory.
    """
    _SEARCHABLE_EXTENSIONS = {".py", ".yaml", ".yml", ".json", ".toml", ".env"}

    searchable_files = [
        (path, content) for path, content in file_contents
        if any(path.endswith(ext) for ext in _SEARCHABLE_EXTENSIONS)
    ]

    resolved = set()
    for sig in error_sigs:
        error_msg = sig.split(":", 1)[1] if ":" in sig else sig

        search_terms: list[str] = []
        for pattern in _ERROR_TERM_PATTERNS:
            matches = re.findall(pattern, error_msg, re.IGNORECASE)
            search_terms.extend(m for m in matches if m.lower() not in _NOISE_WORDS)

        if not search_terms:
            continue

        all_absent = True
        for term in search_terms:
            if len(term) < 4:
                continue
            for _path, content in searchable_files:
                if term in content:
                    all_absent = False
                    break
            if not all_absent:
                break

        if all_absent and search_terms:
            resolved.add(sig)
            _logger.info(
                "Error signature resolved in code: %s (term '%s' absent)",
                sig, search_terms[0],
            )

    return resolved


def _check_errors_against_code(
    error_sigs: set[str], repo_dir: Path
) -> set[str]:
    """Check which error signatures have been fixed in the current codebase.

    Extracts searchable terms from error messages (model names, key names,
    module names) and greps the repo. If the term no longer appears in the
    code, the error is likely resolved.
    """
    resolved = set()
    for sig in error_sigs:
        error_msg = sig.split(":", 1)[1] if ":" in sig else sig

        search_terms: list[str] = []
        for pattern in _ERROR_TERM_PATTERNS:
            matches = re.findall(pattern, error_msg, re.IGNORECASE)
            search_terms.extend(m for m in matches if m.lower() not in _NOISE_WORDS)

        if not search_terms:
            continue

        all_absent = True
        for term in search_terms:
            if len(term) < 4:
                continue
            try:
                result = subprocess.run(
                    ["grep", "-rl", "--include=*.py", "--include=*.yaml",
                     "--include=*.yml", "--include=*.json", "--include=*.toml",
                     "--include=*.env", term, str(repo_dir)],
                    capture_output=True, text=True, timeout=10,
                )
                if result.returncode == 0 and result.stdout.strip():
                    all_absent = False
                    break
            except (subprocess.TimeoutExpired, FileNotFoundError):
                all_absent = False
                break

        if all_absent and search_terms:
            resolved.add(sig)
            _logger.info("Error signature resolved in code: %s (term '%s' absent)", sig, search_terms[0])

    return resolved
