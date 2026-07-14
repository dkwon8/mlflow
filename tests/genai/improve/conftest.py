"""Shared fixtures for improve system tests."""

from __future__ import annotations

import pytest


@pytest.fixture
def make_parsed_trace():
    """Factory for synthetic parsed trace dicts.

    Matches the shape returned by analyzer._parse_trace(), so detectors
    can be tested without MLflow or real trace data.
    """

    def _make(
        trace_id: str = "tr-001",
        tool_names: list[str] | None = None,
        error_count: int = 0,
        error_details: list[dict] | None = None,
        execution_ms: int = 30_000,
        trace_size_bytes: int = 50_000,
        assessments: list[dict] | None = None,
        user_query: str = "test query",
    ) -> dict:
        tool_names = tool_names or []
        tool_counts = {}
        for t in tool_names:
            tool_counts[t] = tool_counts.get(t, 0) + 1

        return {
            "trace_id": trace_id,
            "tool_names": tool_names,
            "unique_tools": set(tool_names),
            "duplicate_tools": {n: c for n, c in tool_counts.items() if c > 1},
            "tool_call_count": len(tool_names),
            "error_count": error_count,
            "error_details": error_details or [],
            "user_query": user_query,
            "execution_ms": execution_ms,
            "trace_size_bytes": trace_size_bytes,
            "assessments": assessments or [],
        }

    return _make
